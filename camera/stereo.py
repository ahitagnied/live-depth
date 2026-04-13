"""Unified stereo camera interface. Camera-agnostic."""

import time
from contextlib import contextmanager


@contextmanager
def open_stereo(camera="oak"):
    """
    Context manager yielding (intrinsics_dict, frames_generator).
    frames_generator yields (left_bgr, right_bgr) tuples.

    Usage:
        with open_stereo("oak") as (intrinsics, frames):
            for left, right in frames:
                ...
    """
    openers = {"oak": _open_oak, "zed": _open_zed}
    if camera not in openers:
        raise ValueError(f"Unknown camera: {camera!r}")
    with openers[camera]() as result:
        yield result


@contextmanager
def _open_oak():
    import depthai as dai
    from camera.oak import (
        init_oak,
        get_camera_intrinsics,
        build_rectification_maps,
        create_stereo_queues,
        rectify_pair,
    )

    device, calib = init_oak()
    try:
        intrinsics = get_camera_intrinsics(calib)
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

            def frames():
                while pipeline.isRunning():
                    fl = fr = None
                    while fl is None or fr is None:
                        if not pipeline.isRunning():
                            return
                        if q_left.has():
                            fl = q_left.get().getCvFrame()
                        if q_right.has():
                            fr = q_right.get().getCvFrame()
                        if fl is None or fr is None:
                            time.sleep(0.005)
                    yield rectify_pair(fl, fr, map1_l, map2_l, map1_r, map2_r)

            yield intrinsics, frames()
    finally:
        device.close()


@contextmanager
def _open_zed():
    from camera.zed import init_zed_camera, get_camera_intrinsics, capture

    zed = init_zed_camera()
    try:
        intrinsics = get_camera_intrinsics(zed)

        def frames():
            while True:
                yield capture(zed)

        yield intrinsics, frames()
    finally:
        zed.close()
