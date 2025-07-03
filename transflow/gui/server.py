import asyncio
import dataclasses
import http.server
import json
import logging
import mimetypes
import multiprocessing
import multiprocessing.queues
import os
import queue
import time
import threading
import tkinter
import traceback
import webbrowser
from pathlib import Path
from tkinter.filedialog import askopenfilename, asksaveasfilename
from urllib.parse import urlparse, parse_qs

import websockets
import websockets.exceptions

logger = logging.getLogger(__name__)
base_dir = Path(__file__).parent


def monitor_job(wss: "WebsocketServer", output_file: str | None, status_queue: multiprocessing.queues.Queue):
    if wss.job is None:
        return
    while wss.job.is_alive():
        try:
            status = status_queue.get(timeout=1)
            wss._broadcast("STATUS " + json.dumps(dataclasses.asdict(status)))
        except queue.Empty:
            continue
        except TimeoutError:
            continue
    while not status_queue.empty():
        status_queue.get(timeout=1)
    status_queue.close()
    wss.job = None
    if output_file is not None:
        wss._broadcast("DONE " + output_file)
    else:
        wss._broadcast("DONE")
    return


class WebsocketServer(threading.Thread):

    def __init__(self, host: str, mjpeg_port: int):
        threading.Thread.__init__(self, daemon=True)
        self.host = host
        self.mjpeg_port = mjpeg_port
        self.port = None
        self.connections = set()
        self.job_cancel_event = None
        self.job = None
        self.job_monitoring = None
        self.output_file = None

    def _broadcast(self, message: str):
        logger.debug("Broadcasting %s", message)
        websockets.broadcast(self.connections, message)

    def _on_client_message(self, websocket, message: str):
        logger.debug("Websocket %s \"%s\"", websocket.id.hex, message)
        if message == "PONG":
            return
        cmd, args_string = None, None
        if " " in message:
            cmd, args_string = message.split(" ", 1)
        else:
            cmd = message
        args = json.loads(args_string) if args_string else {}
        if cmd == "FILE_OPEN":
            window = tkinter.Tk()
            window.wm_attributes("-topmost", 1)
            window.withdraw()
            filename = askopenfilename(
                parent=window,
                initialdir=(base_dir / ".." / ".." / "assets").as_posix(),
                filetypes=[("Allowed types", args["filetypes"])])
            if filename != "":
                self._broadcast(f"FILE {args['key']} {filename}")
        elif cmd == "FILE_SAVE":
            window = tkinter.Tk()
            window.wm_attributes("-topmost", 1)
            window.withdraw()
            filename = asksaveasfilename(
                parent=window,
                initialdir=(base_dir / ".." / ".." / "out").as_posix(),
                defaultextension=args["defaultextension"],
                filetypes=[("Allowed types", args["filetypes"])],
                initialfile=f"transflow-{time.time():.0f}"
            )
            if filename != "":
                self._broadcast(f"FILE {args['key']} {filename}")
        elif cmd == "GENERATE":
            from ..pipeline import transfer, Config
            from ..utils import parse_timestamp
            print("Job args:")
            print(args)
            seek_time = parse_timestamp(args["flowSource"]["seekTime"]) if args["flowSource"]["seekTime"] is not None else 0
            duration_time = parse_timestamp(args["flowSource"]["durationTime"])
            bitmap_path = None
            if args["bitmapSource"]["type"] == "file":
                bitmap_path = args["bitmapSource"]["file"]
            elif args["bitmapSource"]["type"] == "color":
                bitmap_path = args["bitmapSource"]["color"]
            else:
                bitmap_path = args["bitmapSource"]["type"]
            initial_canvas = None
            if args["accumulator"]["initialCanvasFile"] is None:
                initial_canvas = args["accumulator"]["initialCanvasColor"]
            else:
                initial_canvas = args["accumulator"]["initialCanvasFile"]
            output_paths = [f"mjpeg:{self.mjpeg_port}:{self.host}"]
            if args["output"]["file"] is not None:
                output_paths.append(args["output"]["file"])
                self.output_file = args["output"]["file"]
            else:
                self.output_file = None
            config = Config(
                args["flowSource"]["file"],
                bitmap_path,
                output_paths,
                None,
                execute=False,
                replace=False,
                safe=True,
                seed=args["seed"],
                use_mvs=args["flowSource"]["useMvs"],
                direction=args["flowSource"]["direction"],
                acc_method=args["accumulator"]["method"],
                mask_path=args["flowSource"]["maskPath"],
                kernel_path=args["flowSource"]["kernelPath"],
                cv_config=args["flowSource"]["cvConfig"],
                flow_filters=args["flowSource"]["flowFilters"],
                reset_mode=args["accumulator"]["resetMode"],
                reset_alpha=args["accumulator"]["resetAlpha"],
                reset_mask_path=args["accumulator"]["resetMask"],
                heatmap_mode=args["accumulator"]["heatmapMode"],
                heatmap_args=args["accumulator"]["heatmapArgs"],
                heatmap_reset_threshold=args["accumulator"]["heatmapResetThreshold"],
                accumulator_background=args["accumulator"]["background"],
                stack_composer=args["accumulator"]["stackComposer"],
                initial_canvas=initial_canvas,
                bitmap_mask_path=args["accumulator"]["bitmapMask"],
                crumble=args["accumulator"]["crumble"],
                bitmap_alteration_path=args["bitmapSource"]["alterationPath"],
                preview_output=False,
                vcodec=args["output"]["vcodec"],
                round_flow=args["flowSource"]["roundFlow"],
                export_flow=args["flowSource"]["exportFlow"],
                output_intensity=args["output"]["outputIntensity"],
                output_heatmap=args["output"]["outputHeatmap"],
                output_accumulator=args["output"]["outputAccumulator"],
                render_scale=args["output"]["renderScale"],
                render_colors=args["output"]["renderColors"],
                render_binary=args["output"]["renderBinary"],
                checkpoint_every=args["output"]["checkpointEvery"],
                checkpoint_end=args["output"]["checkpointEnd"],
                seek_time=seek_time,
                duration_time=duration_time,
                repeat=args["flowSource"]["repeat"],
                bitmap_seek_time=parse_timestamp(args["bitmapSource"]["seekTime"]),
                bitmap_repeat=args["bitmapSource"]["repeat"],
                bitmap_introduction_flags=args["accumulator"]["bitmapIntroductionFlags"],
                lock_mode=args["flowSource"]["lockMode"],
                lock_expr=args["flowSource"]["lockExpr"],
            )
            self.job_cancel_event = threading.Event()
            status_queue = multiprocessing.Queue(maxsize=1)
            self.job = threading.Thread(
                target=transfer,
                args=[config],
                kwargs={
                    "cancel_event": self.job_cancel_event,
                    "status_queue": status_queue})
            self.job.start()
            self._broadcast(f"PREVIEW http://{self.host}:{self.mjpeg_port}/transflow")
            self.job_monitoring = threading.Thread(target=monitor_job, args=(self, args["output"]["file"], status_queue))
            self.job_monitoring.start()
        elif cmd == "INTERRUPT":
            if self.job is None or self.job_cancel_event is None or self.job_monitoring is None:
                return
            self.job_cancel_event.set()
            self.job_monitoring.join()
            self._broadcast("CANCEL")
        elif cmd == "RELOAD":
            self._broadcast(f"""RELOAD {json.dumps({
                "ongoing": self.job is not None,
                "outputFile": self.output_file,
                "previewUrl": f"http://{self.host}:{self.mjpeg_port}/transflow",
            })}""")

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
                        traceback.print_exc()

            except websockets.exceptions.ConnectionClosedError:
                pass
            except ConnectionResetError:
                pass
            logger.debug("WebSocket client disconnected: %s", websocket.id.hex)
            self.connections.remove(websocket)
        async def start_server():
            async with websockets.serve(register, self.host, None) as wserver:
                self.port = list(wserver.sockets)[0].getsockname()[1]
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
                return self.serve_file_range(media_path)
            else:
                return self.send_error(404, "File not found")
        elif parsed.path == "/wss":
            assert isinstance(self.server, "WebServer")
            return self.serve_text(f"ws://{self.server.wss.host}:{self.server.wss.port}")
        return super().do_GET()

    def serve_text(self, text: str, encoding: str = "utf8"):
        self.send_response(200)
        self.send_header('Content-Type', "text/plain")
        self.end_headers()
        self.wfile.write(text.encode(encoding))

    def serve_file_range(self, path):
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


class WebServer(http.server.HTTPServer):

    def __init__(self, host: str, port: int, mjpeg_port: int):
        http.server.HTTPServer.__init__(self, (host, port), WebHandler)
        self.wss = WebsocketServer(host, mjpeg_port)
        self.wss.start()


def start_gui(host: str = "localhost", port: int = 8000, mjpeg_port: int = 8001):
    with WebServer(host, port, mjpeg_port) as httpd:
        url = f"http://{host}:{port}"
        logger.info(f"Listening at {url}")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
