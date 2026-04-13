"""
MJPEG-over-HTTP streaming server. Camera-agnostic.

Usage: pass any iterable of BGR numpy frames to stream().
"""

import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

_cond = threading.Condition()
_jpg: bytes = b""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with _cond:
                    _cond.wait()
                    jpg = _jpg
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
        except (BrokenPipeError, ConnectionResetError):
            pass


def stream(frames, port=8080):
    """
    Serve an iterable of BGR numpy frames as MJPEG over HTTP.
    Blocks until the iterable is exhausted or KeyboardInterrupt.
    """
    global _jpg

    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"MJPEG streaming on :{port}")

    try:
        n, t0 = 0, time.monotonic()
        for frame in frames:
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with _cond:
                _jpg = jpg.tobytes()
                _cond.notify_all()

            n += 1
            elapsed = time.monotonic() - t0
            if elapsed >= 1.0:
                sys.stdout.write(f"\rfps: {n / elapsed:.1f}   ")
                sys.stdout.flush()
                n, t0 = 0, time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
