import http.server
import json
import os
import sys
import threading
import unittest
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from transflow.gui.server import WebServer


class TestGui(unittest.TestCase):

    HOST = "localhost"
    PORT = 8000
    MJPEG_PORT = 8001

    def setUp(self):
        http.server.HTTPServer.allow_reuse_address = True
        http.server.BaseHTTPRequestHandler.log_message = lambda *a, **kw: None
        self.server = WebServer(self.HOST, self.PORT, self.MJPEG_PORT)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.addCleanup(self._stop_server)

    def test_basic_request(self):
        url = f"http://{self.HOST}:{self.PORT}/ping"
        with urllib.request.urlopen(url, timeout=5) as r:
            body = r.read()
        self.assertEqual(body, b"PONG")
    
    def test_generate(self):
        class Namespace:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        dummy = Namespace(id=Namespace(hex="foo"))
        config = {
            "seed": 1,
            "flowSource": {
                "file": "assets/River.mp4",
                "direction": "backward",
                "maskPath": None,
                "kernelPath": None,
                "cvConfig": None,
                "flowFilters": None,
                "useMvs": False,
                "roundFlow": False,
                "exportFlow": False,
                "seekTime": None,
                "durationTime": "00:00:00.000",
                "repeat": 1,
                "lockMode": "stay",
                "lockExpr": None,
            },
            "compositor": {
                "layerCount": 1,
                "layers": [{
                    "classname": "moveref",
                    "maskAlpha": None,
                    "maskSource": None,
                    "maskDestination": None,
                    "flagMoveTransparent": False,
                    "flagMoveToEmpty": True,
                    "flagMoveToFilled": True,
                    "flagLeaveEmpty": False,
                    "introduceEmpty": True,
                    "introduceFilled": True,
                    "introduceMoving": True,
                    "introduceUnmoving": True,
                    "introduceOnce": False,
                    "introduceAllEmpty": False,
                    "introduceAllFilled": False,
                    "resetMode": "off",
                    "maskReset": None,
                    "resetRandomFactor": 0.1,
                    "resetConstantStep": 1,
                    "resetLinearFactor": 0.1,
                    "resetSource": False,
                    "sourceCount": 1,
                    "sources": [{
                        "file": None,
                        "type": "bwnoise",
                        "color": "#cff010",
                        "alterationPath": None,
                        "maskIntroduction": None,
                        "seekTime": None,
                        "repeat": 1, 
                    }]
                }],
                "backgroundColor": "#ffffff"
            },
            "output": {
                "file": None,
                "viewFlow": False,
                "viewFlowMagnitude": False,
                "renderScale": 0.1,
                "renderColors": None,
                "renderBinary": False,
                "checkpointEvery": None,
                "checkpointEnd": False,
                "vcodec": "h264",
            }
        }
        self.server.wss._on_client_message(dummy, "GENERATE " + json.dumps(config))

    def _stop_server(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()


if __name__ == "__main__":
    unittest.main()
