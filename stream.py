# python stream.py [--camera zed|oak] [--port 8080] [--scale 0.5]
# Open: http://<jetson-ip>:8080

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from camera.stereo import open_stereo
from main import DepthEstimator, WEIGHTS_PATH

_cond = threading.Condition()
_jpg: bytes = b""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b'<html><body style="margin:0;background:#000">'
                b'<img src="/stream" style="width:100%;height:100vh;object-fit:contain">'
                b'</body></html>'
            )
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with _cond:
                        _cond.wait()
                        jpg = _jpg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera",      default="zed", choices=["zed", "oak"])
    ap.add_argument("--port",        type=int,   default=8080)
    ap.add_argument("--weights",     default=str(WEIGHTS_PATH))
    ap.add_argument("--scale",       type=float, default=1.0)
    ap.add_argument("--valid-iters", type=int,   default=8)
    ap.add_argument("--max-disp",    type=int,   default=192)
    args = ap.parse_args()

    global _jpg

    est = DepthEstimator(args.weights, args.scale, args.valid_iters, args.max_disp)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"http://0.0.0.0:{args.port}")

    try:
        with open_stereo(args.camera) as (_, frames):
            n, t0 = 0, time.monotonic()
            for left, right in frames:
                vis = est.infer(left, right)
                _, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
                with _cond:
                    _jpg = jpg.tobytes()
                    _cond.notify_all()
                n += 1
                if (elapsed := time.monotonic() - t0) >= 1.0:
                    sys.stdout.write(f"\rfps {n/elapsed:.1f}   ")
                    sys.stdout.flush()
                    n, t0 = 0, time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
