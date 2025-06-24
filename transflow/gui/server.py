import asyncio
import http.server
import json
import logging
import mimetypes
import os
import threading
import tkinter
import webbrowser
from pathlib import Path
from tkinter.filedialog import askopenfilename
from urllib.parse import urlparse, parse_qs

import websockets
import websockets.exceptions

logger = logging.getLogger(__name__)
base_dir = Path(__file__).parent

class WebsocketServer(threading.Thread):

    def __init__(self, host: str, mjpeg_port: int):
        threading.Thread.__init__(self, daemon=True)
        self.host = host
        self.mjpeg_port = mjpeg_port
        self.port = None
        self.connections = set()
    
    def _broadcast(self, message: str):
        logger.debug("Broadcasting %s", message)
        websockets.broadcast(self.connections, message)

    def _on_client_message(self, websocket, message: str):
        logger.debug("Websocket %s \"%s\"", websocket.id.hex, message)
        if message == "PONG":
            return
        elif message.startswith("FILEIN "):
            window = tkinter.Tk()
            window.wm_attributes("-topmost", 1)
            window.withdraw()
            filename = askopenfilename(parent=window, initialdir=(base_dir / ".." / ".." / "assets").as_posix(), filetypes=[("video files", "*.mp4")])
            if filename != "":
                self._broadcast(f"{message} {filename}")
            return
        elif message.startswith("GEN "):
            config = json.loads(message[4:])
            from ..pipeline import transfer
            output_path = f"mjpeg:{self.mjpeg_port}:{self.host}"
            mjpeg_url = f"http://{self.host}:{self.mjpeg_port}/transflow"
            self._broadcast(f"OUT {mjpeg_url}")
            transfer(config["flowSource"]["file"], config["bitmapSource"]["file"], output_path, None, acc_method=config["accumulator"]["method"])

    def run(self):
        async def register(websocket):
            logger.debug("WebSocket client connected: %s", websocket.id.hex)
            self.connections.add(websocket)
            try:
                async for message in websocket:
                    try:
                        self._on_client_message(websocket, message)
                    except Exception as err:
                        logger.error(
                            "Error on client %s message \"%s\": %s",
                            websocket.id.hex,
                            message,
                            err)
            except websockets.exceptions.ConnectionClosedError:
                pass
            except ConnectionResetError:
                pass
            logger.debug("WebSocket client disconnected: %s", websocket.id.hex)
            self.connections.remove(websocket)
        async def start_server():
            async with websockets.serve(register, self.host, None) as wserver:
                self.port = wserver.sockets[0].getsockname()[1]
                logger.info("Starting websocket server at ws://%s:%d", self.host, self.port)
                await asyncio.Future()
        asyncio.run(start_server())


class WebHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(__file__), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/media":
            query = parse_qs(parsed.query)
            media_path = query.get('url', [None])[0]
            if media_path and os.path.isfile(media_path):
                # TODO: how about image files?
                self.serve_video_file(media_path)
            else:
                self.send_error(404, "File not found")
        return super().do_GET()
    
    def serve_video_file(self, path):
        file_size = os.path.getsize(path)
        mime_type, _ = mimetypes.guess_type(path)
        mime_type = mime_type or 'application/octet-stream'

        range_header = self.headers.get('Range')
        if range_header:
            range_value = range_header.strip().split('=')[1]
            range_start, range_end = range_value.split('-')
            start = int(range_start)
            end = int(range_end) if range_end else file_size - 1
            length = end - start + 1

            self.send_response(206)
            self.send_header('Content-Type', mime_type)
            self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
            self.send_header('Content-Length', str(length))
            self.send_header('Accept-Ranges', 'bytes')
            self.end_headers()

            with open(path, 'rb') as f:
                f.seek(start)
                self.wfile.write(f.read(length))
        else:
            self.send_response(200)
            self.send_header('Content-Type', mime_type)
            self.send_header('Content-Length', str(file_size))
            self.send_header('Accept-Ranges', 'bytes')
            self.end_headers()
            with open(path, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
    
    def end_headers(self):
        self.send_header("Wss-Host", str(self.server.wss.host))
        self.send_header("Wss-Port", str(self.server.wss.port))
        return super().end_headers()


class WebServer(http.server.HTTPServer):

    def __init__(self, host: str, port: int, mjpeg_port: int):
        http.server.HTTPServer.__init__(self, (host, port), WebHandler)
        self.wss = WebsocketServer(host, mjpeg_port)
        self.wss.start()


def start_gui(host: str = "localhost", port: int = 8000, mjpeg_port: int = 8080):
    with WebServer(host, port, mjpeg_port) as httpd:
        url = f"http://{host}:{port}"
        logger.info(f"Listening at {url}")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
