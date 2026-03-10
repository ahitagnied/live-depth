"""
python stream.py --near 300 --far 4000 --port 9090

Open in browser:  http://<jetson-ip>:8080
"""

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import depthai as dai
import numpy as np

from oak import util

WIDTH  = util.WIDTH
HEIGHT = util.HEIGHT

_frame_lock = threading.Lock()
_latest_jpg: bytes = b""


def _colorize(depth_mm: np.ndarray, near: int, far: int) -> np.ndarray:
    clipped = np.clip(depth_mm, near, far).astype(np.float32)
    norm    = ((clipped - near) / (far - near) * 255).astype(np.uint8)
    return cv2.applyColorMap(255 - norm, cv2.COLORMAP_JET)


def capture_loop(near: int, far: int) -> None:
    global _latest_jpg

    device   = dai.Device()
    pipeline = dai.Pipeline(device)

    cam_l = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    cam_r = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetType.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)
    stereo.setOutputSize(WIDTH, HEIGHT)

    cam_l.requestOutput((WIDTH, HEIGHT), type=dai.ImgFrame.Type.GRAY8).link(stereo.left)
    cam_r.requestOutput((WIDTH, HEIGHT), type=dai.ImgFrame.Type.GRAY8).link(stereo.right)

    q = stereo.depth.createOutputQueue()

    with pipeline:
        pipeline.start()
        depth_mm = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
        print(f"OAK {device.getMxId()} streaming  near={near}mm  far={far}mm")

        while pipeline.isRunning():
            frame = q.tryGet()
            if frame is not None:
                depth_mm = frame.getCvFrame()

            vis = _colorize(depth_mm, near, far)
            ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with _frame_lock:
                    _latest_jpg = jpg.tobytes()

            time.sleep(0.016)  # ~60 fps cap; OAK runs at 30 fps anyway

    device.close()


class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # silence per-request logs

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
                <html><body style="margin:0;background:#000">
                <img src="/stream" style="width:100%;height:100vh;object-fit:contain">
                </body></html>
            """)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with _frame_lock:
                        jpg = _latest_jpg
                    if jpg:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                        )
                    time.sleep(0.033)  # ~30 fps to client
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--near", type=int, default=200)
    ap.add_argument("--far",  type=int, default=6000)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    t = threading.Thread(target=capture_loop, args=(args.near, args.far), daemon=True)
    t.start()

    # wait for first frame before advertising the URL
    while not _latest_jpg:
        time.sleep(0.05)

    print(f"http://0.0.0.0:{args.port}  (open on any device on this network)")
    HTTPServer(("0.0.0.0", args.port), MJPEGHandler).serve_forever()


if __name__ == "__main__":
    main()
