import json
import os

import cv2
import depthai as dai
import numpy as np

WIDTH = 640
HEIGHT = 480


def build_rectification_maps(calib: dai.CalibrationHandler):
    M_left = np.array(calib.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_B, WIDTH, HEIGHT
    ))
    M_right = np.array(calib.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_C, WIDTH, HEIGHT
    ))

    D_left = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B))
    D_right = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C))

    extrinsics = np.array(calib.getCameraExtrinsics(
        dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C
    ))
    R = extrinsics[:3, :3]
    T = extrinsics[:3, 3]

    img_size = (WIDTH, HEIGHT)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        M_left, D_left,
        M_right, D_right,
        img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    map1_left, map2_left = cv2.initUndistortRectifyMap(
        M_left, D_left, R1, P1, img_size, cv2.CV_32FC1
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        M_right, D_right, R2, P2, img_size, cv2.CV_32FC1
    )

    fx = float(P1[0, 0])
    fy = float(P1[1, 1])
    cx = float(P1[0, 2])
    cy = float(P1[1, 2])
    baseline_m = abs(float(P2[0, 3]) / float(P2[0, 0])) / 100.0

    K_rect = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "baseline": baseline_m}

    return map1_left, map2_left, map1_right, map2_right, K_rect


def save_rectified(
    frame_left: np.ndarray,
    frame_right: np.ndarray,
    map1_l, map2_l, map1_r, map2_r,
    K_rect: dict,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    rect_left = cv2.remap(frame_left, map1_l, map2_l, cv2.INTER_LINEAR)
    rect_right = cv2.remap(frame_right, map1_r, map2_r, cv2.INTER_LINEAR)

    left_bgr = cv2.cvtColor(rect_left, cv2.COLOR_GRAY2BGR)
    right_bgr = cv2.cvtColor(rect_right, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(out_dir, "left.png"), left_bgr)
    cv2.imwrite(os.path.join(out_dir, "right.png"), right_bgr)

    with open(os.path.join(out_dir, "intrinsics.json"), "w") as f:
        json.dump({**K_rect, "width": WIDTH, "height": HEIGHT}, f, indent=2)

    cv2.imwrite(os.path.join(out_dir, "mask.png"), np.full((HEIGHT, WIDTH), 255, dtype=np.uint8))
    print(f"Captured to {out_dir}/ (fx={K_rect['fx']:.1f} baseline={K_rect['baseline']:.4f}m)")
