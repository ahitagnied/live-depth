import os
import time

import cv2
import depthai as dai
import numpy as np

from . import util


def init_oak():
    device = dai.Device()
    calib = device.readCalibration()
    return device, calib


def create_stereo_queues(pipeline):
    cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    q_left = cam_left.requestOutput(
        (util.WIDTH, util.HEIGHT), type=dai.ImgFrame.Type.GRAY8
    ).createOutputQueue()
    cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    q_right = cam_right.requestOutput(
        (util.WIDTH, util.HEIGHT), type=dai.ImgFrame.Type.GRAY8
    ).createOutputQueue()
    return q_left, q_right


def rectify_pair(frame_left, frame_right, map1_l, map2_l, map1_r, map2_r):
    left_bgr = cv2.cvtColor(
        cv2.remap(frame_left, map1_l, map2_l, cv2.INTER_LINEAR), cv2.COLOR_GRAY2BGR
    )
    right_bgr = cv2.cvtColor(
        cv2.remap(frame_right, map1_r, map2_r, cv2.INTER_LINEAR), cv2.COLOR_GRAY2BGR
    )
    return left_bgr, right_bgr


def get_camera_intrinsics(calib: dai.CalibrationHandler) -> dict:
    _, _, _, _, K_rect = util.build_rectification_maps(calib)
    return {
        "fx": K_rect["fx"],
        "fy": K_rect["fy"],
        "cx": K_rect["cx"],
        "cy": K_rect["cy"],
        "baseline": K_rect["baseline"],
        "width": util.WIDTH,
        "height": util.HEIGHT,
    }


def capture_stereo(device: dai.Device):
    calib = device.readCalibration()
    map1_l, map2_l, map1_r, map2_r, _ = util.build_rectification_maps(calib)

    with dai.Pipeline(device) as pipeline:
        q_left, q_right = create_stereo_queues(pipeline)
        pipeline.start()

        frame_left_raw = None
        frame_right_raw = None
        for _ in range(300):
            if q_left.has():
                frame_left_raw = q_left.get().getCvFrame()
            if q_right.has():
                frame_right_raw = q_right.get().getCvFrame()
            if frame_left_raw is not None and frame_right_raw is not None:
                break
            time.sleep(0.016)

    if frame_left_raw is None or frame_right_raw is None:
        raise RuntimeError("OAK: failed to get one frame from both cameras")

    return rectify_pair(frame_left_raw, frame_right_raw, map1_l, map2_l, map1_r, map2_r)


def capture_rectified(out_dir: str = "tmp") -> None:
    os.makedirs(out_dir, exist_ok=True)
    device = dai.Device()
    calib = device.readCalibration()

    map1_l, map2_l, map1_r, map2_r, K_rect = util.build_rectification_maps(calib)
    print(f"Rectified intrinsics: {K_rect}")

    with dai.Pipeline(device) as pipeline:
        q_left, q_right = create_stereo_queues(pipeline)
        pipeline.start()
        print("Streaming. Press 'c' to capture, 'q' to quit.")

        frame_left = np.zeros((util.HEIGHT, util.WIDTH), dtype=np.uint8)
        frame_right = np.zeros((util.HEIGHT, util.WIDTH), dtype=np.uint8)

        while pipeline.isRunning():
            if q_left.has():
                frame_left = q_left.get().getCvFrame()
            if q_right.has():
                frame_right = q_right.get().getCvFrame()

            left_bgr, right_bgr = rectify_pair(frame_left, frame_right, map1_l, map2_l, map1_r, map2_r)
            stereo_view = np.hstack([left_bgr, right_bgr])

            for y in range(0, util.HEIGHT, 40):
                cv2.line(stereo_view, (0, y), (util.WIDTH * 2, y), (0, 255, 0), 1)

            cv2.putText(
                stereo_view, "Rectified L|R  [c] capture  [q] quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.imshow("OAK-D Lite — rectified stereo", stereo_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                util.save_rectified(
                    frame_left.copy(), frame_right.copy(),
                    map1_l, map2_l, map1_r, map2_r, K_rect,
                    out_dir,
                )

    device.close()
