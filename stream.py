# python stream.py [--port 8080] [--scale 0.5]
# Open: http://<jetson-ip>:8080

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2

from camera.oak import init_oak, get_camera_intrinsics, build_rectification_maps, create_stereo_queues, rectify_pair
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
    import depthai as dai

    ap = argparse.ArgumentParser()
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

    device, calib = init_oak()
    try:
        map1_l, map2_l, map1_r, map2_r, _ = build_rectification_maps(calib)

        with dai.Pipeline(device) as pipeline:
            q_left, q_right = create_stereo_queues(pipeline)
            pipeline.start()

            # let auto-exposure settle, then flush stale frames
            time.sleep(5)
            while q_left.has():
                q_left.get()
            while q_right.has():
                q_right.get()

            n, t0 = 0, time.monotonic()
            try:
                while pipeline.isRunning():
                    fl = fr = None
                    while fl is None or fr is None:
                        if not pipeline.isRunning():
                            break
                        if q_left.has():
                            fl = q_left.get().getCvFrame()
                        if q_right.has():
                            fr = q_right.get().getCvFrame()
                        if fl is None or fr is None:
                            time.sleep(0.005)
                    if fl is None or fr is None:
                        break
                    left, right = rectify_pair(fl, fr, map1_l, map2_l, map1_r, map2_r)
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
        device.close()
        server.shutdown()


if __name__ == "__main__":
    main()
