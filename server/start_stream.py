#!/usr/bin/env python3
"""
Live RGB stream via MJPEG over HTTP.

Jetson: ./server/stream.sh --camera oak
        ./server/stream.sh --camera zed
Local:    open http://<jetson-tailscale-ip>:8080 in browser
        or: python server/view_stream.py <ip>
"""

import argparse

from server.mjpeg import stream

WIDTH, HEIGHT = 640, 480


def _oak_frames():
    import depthai as dai
    from camera.oak import init_oak

    device, _ = init_oak()
    try:
        with dai.Pipeline(device) as pipeline:
            cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            q = cam.requestOutput(
                (WIDTH, HEIGHT), type=dai.ImgFrame.Type.BGR888p
            ).createOutputQueue()
            pipeline.start()
            while pipeline.isRunning():
                yield q.get().getCvFrame()
    finally:
        device.close()


def _zed_frames():
    from camera.stereo import open_stereo

    with open_stereo("zed") as (_, frames):
        for left, _ in frames:
            yield left


CAMERAS = {"oak": _oak_frames, "zed": _zed_frames}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=CAMERAS, default="oak")
    args = parser.parse_args()
    stream(CAMERAS[args.camera]())
