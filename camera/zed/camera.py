import cv2
import pyzed.sl as sl


def init_zed_camera(resolution=sl.RESOLUTION.HD1200):
    """
    Initialize ZED camera with given resolution.

    Args:
        resolution: sl.RESOLUTION - camera resolution

    Returns:
        zed: sl.Camera - ZED camera object
    """
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = resolution
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.coordinate_units = sl.UNIT.METER

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Failed to open ZED camera")

    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, -1)

    runtime = sl.RuntimeParameters()
    for _ in range(30):
        zed.grab(runtime)

    return zed


def get_camera_intrinsics(zed):
    """
    Get camera intrinsics needed to convert depth map to point cloud.

    Returns:
        dict with keys:
            - 'fx': float - focal length x
            - 'fy': float - focal length y
            - 'cx': float - principal point x
            - 'cy': float - principal point y
            - 'baseline': float - stereo baseline in meters
            - 'width': int - image width
            - 'height': int - image height
    """
    calibration_params = (
        zed.get_camera_information().camera_configuration.calibration_parameters
    )
    left_cam = calibration_params.left_cam
    resolution = zed.get_camera_information().camera_configuration.resolution
    baseline = calibration_params.get_camera_baseline()

    return {
        "fx": left_cam.fx,
        "fy": left_cam.fy,
        "cx": left_cam.cx,
        "cy": left_cam.cy,
        "baseline": baseline,
        "width": resolution.width,
        "height": resolution.height,
    }


def capture(zed):
    """
    One grab + retrieve; returns (frame_left, frame_right) as numpy BGR. In memory only.
    """
    runtime = sl.RuntimeParameters()
    if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("ZED grab failed")

    rgb_left_mat = sl.Mat()
    rgb_right_mat = sl.Mat()

    zed.retrieve_image(rgb_left_mat, sl.VIEW.LEFT)
    zed.retrieve_image(rgb_right_mat, sl.VIEW.RIGHT)

    frame_left = cv2.cvtColor(rgb_left_mat.get_data(), cv2.COLOR_BGRA2BGR)
    frame_right = cv2.cvtColor(rgb_right_mat.get_data(), cv2.COLOR_BGRA2BGR)

    return frame_left, frame_right
