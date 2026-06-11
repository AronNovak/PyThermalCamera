#!/usr/bin/env python3
"""
thermal_bridge.py - Serve the thermal camera as an MJPEG-over-HTTP stream.

Re-exposes an InfiRay/Topdon thermal camera (parsed + colormapped by thermalcam)
as a plain MJPEG stream that ANY app can open like a webcam - no kernel module,
no root, no reboot, no code dependency:

    python3 src/thermal_bridge.py --colormap inferno --rotate 90

    # then, in any consumer:
    OpenCV:   cv2.VideoCapture("http://127.0.0.1:8090/stream.mjpg")
    ffmpeg:   ffmpeg -i http://127.0.0.1:8090/stream.mjpg ...
    browser:  http://127.0.0.1:8090/      (preview page)

This exists because the camera's default UVC output is a torn/garbage frame; the
bridge does the 512x484 radiometric parse and hands out a clean thermal image.
"""

import argparse
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

import thermalcam

BOUNDARY = "frame"

# Latest encoded JPEG, published by the capture thread and consumed by clients.
_latest = {"jpeg": None}
_new_frame = threading.Condition()
_stop = threading.Event()


def capture_loop(app, out_size, quality):
    """Grab frames, colormap them, publish the newest JPEG. Skips desynced/tiled
    frames, and reopens the camera if the stream wedges into a stuck desync."""
    encode = [cv2.IMWRITE_JPEG_QUALITY, quality]
    skips = 0
    while not _stop.is_set():
        ok, frame = app.cap.read()
        raw = app._as_raw(frame) if ok else None
        bgr = app.colormap_frame(raw) if raw is not None else None
        if bgr is None:               # bad frame; keep serving the previous one
            skips += 1
            if skips % 60 == 0:       # ~2-4 s of nothing usable -> recover
                print(f"[bridge] {skips} unusable frames, reopening camera...", flush=True)
                app.reopen()
            continue
        skips = 0
        if out_size is not None:
            bgr = cv2.resize(bgr, out_size, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", bgr, encode)
        if not ok:
            continue
        with _new_frame:
            _latest["jpeg"] = buf.tobytes()
            _new_frame.notify_all()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):        # quiet
        pass

    def do_GET(self):
        if self.path.startswith("/stream") or self.path.rstrip("/") == "":
            if self.path.rstrip("/") == "":
                return self._page()
            return self._stream()
        if self.path.startswith("/snapshot"):
            return self._snapshot()
        self.send_error(404)

    def _page(self):
        html = (b"<html><body style='margin:0;background:#111'>"
                b"<img src='/stream.mjpg' style='width:100%'></body></html>")
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _snapshot(self):
        with _new_frame:
            _new_frame.wait(timeout=2.0)
            jpeg = _latest["jpeg"]
        if jpeg is None:
            return self.send_error(503)
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.end_headers()
        self.wfile.write(jpeg)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while not _stop.is_set():
                with _new_frame:
                    _new_frame.wait(timeout=2.0)
                    jpeg = _latest["jpeg"]
                if jpeg is None:
                    continue
                self.wfile.write(b"--" + BOUNDARY.encode() + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def build_app(args):
    """Open the camera through thermalcam with the requested settings."""
    names = [n.lower() for _c, n in thermalcam.COLORMAPS]
    if args.colormap.lower() not in names:
        sys.exit(f"--colormap must be one of: {', '.join(names)}")

    tc_argv = ["--scale", "1", "--rotate", str(args.rotate), "--flip", args.flip,
               "--temp-scale", args.temp_scale, "--temp-order", args.temp_order]
    if args.device:
        tc_argv += ["--device", args.device]
    if args.resolution:
        tc_argv += ["--resolution", args.resolution]
    if args.no_destripe:
        tc_argv += ["--no-destripe"]
    tc_argv += ["--smooth", str(args.smooth)]

    app = thermalcam.ThermalApp(thermalcam.parse_args(tc_argv))
    app.colormap = names.index(args.colormap.lower())
    app.open_camera()
    return app


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="Bind address (0.0.0.0 to expose on the LAN).")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--colormap", default="inferno", help="Palette (jet, inferno, hot, ...). Default inferno.")
    p.add_argument("--width", type=int, help="Output width (default: camera native).")
    p.add_argument("--height", type=int, help="Output height (default: camera native).")
    p.add_argument("--quality", type=int, default=85, help="JPEG quality 1-100.")
    p.add_argument("--device", help="Video device. Default: auto-detect.")
    p.add_argument("--resolution", help="Force capture WxH.")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    p.add_argument("--flip", default="none", choices=["none", "h", "v"])
    p.add_argument("--temp-scale", default="auto")
    p.add_argument("--temp-order", default="le", choices=["le", "be"])
    p.add_argument("--no-destripe", action="store_true")
    p.add_argument("--smooth", type=float, default=0.5)
    args = p.parse_args(argv)

    app = build_app(args)
    out_size = (args.width, args.height) if args.width and args.height else None

    worker = threading.Thread(target=capture_loop, args=(app, out_size, args.quality), daemon=True)
    worker.start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Thermal MJPEG stream at {url}/stream.mjpg  (preview: {url}/  snapshot: {url}/snapshot.jpg)", flush=True)
    print("Open it from any app, e.g.:", flush=True)
    print(f"  python3 -c \"import cv2; c=cv2.VideoCapture('{url}/stream.mjpg'); print(c.read()[0])\"", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _stop.set()
        server.shutdown()
        app.cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
