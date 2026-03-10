"""
Run Fast-FoundationStereo locally on Jetson (or any CUDA machine) — no Modal, no network.

Setup on Jetson (JetPack 6, CUDA 12):
    pip install torch torchvision --index-url <NVIDIA JetPack torch URL>
    pip install timm einops omegaconf scipy numpy scikit-image \
                opencv-contrib-python-headless imageio pyyaml open3d
    # xformers: compile from source or skip — FFS falls back to vanilla attention

Usage:
    python main.py --left L.png --right R.png --intrinsics K.json
    python main.py --left L.png --right R.png --weights /data/weights/23-36-37 --scale 0.5
"""

import argparse
import json
import sys
import time
from pathlib import Path


FFS_PATH     = Path(__file__).parent / "external" / "Fast-FoundationStereo"
WEIGHTS_PATH = Path(__file__).parent / "external" / "Fast-FoundationStereo" / "weights" / "23-36-37"


def load_model(weights: Path):
    import torch
    import yaml

    ckpt = weights / "model_best_bp2_serialize.pth"
    if not ckpt.exists():
        sys.exit(f"weights not found: {ckpt}")

    with open(weights / "cfg.yaml") as f:
        cfg = yaml.safe_load(f)

    model = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    model.cuda().eval()
    return model, cfg


def infer(model, left_path: Path, right_path: Path, intrinsics_txt: str | None,
          scale: float, valid_iters: int, max_disp: int, zfar: float) -> dict:
    import cv2
    import imageio.v3 as iio
    import numpy as np
    import torch

    sys.path.insert(0, str(FFS_PATH))
    from core.utils.utils import InputPadder
    from Utils import vis_disparity, depth2xyzmap, toOpen3dCloud, o3d

    try:
        from Utils import AMP_DTYPE
    except ImportError:
        import torch
        AMP_DTYPE = torch.float16

    model.args.valid_iters = valid_iters
    model.args.max_disp    = max_disp

    def decode(p: Path) -> np.ndarray:
        img = cv2.imread(str(p))
        if img is None:
            sys.exit(f"could not read {p}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img0 = decode(left_path)
    img1 = decode(right_path)

    if scale != 1.0:
        img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
        img1 = cv2.resize(img1, dsize=(img0.shape[1], img0.shape[0]))

    H, W    = img0.shape[:2]
    orig0   = img0.copy()
    orig1   = img1.copy()

    t0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    t1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)

    padder  = InputPadder(t0.shape, divis_by=32, force_square=False)
    t0, t1  = padder.pad(t0, t1)

    torch.autograd.set_grad_enabled(False)
    with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        disp = model.forward(t0, t1, iters=valid_iters, test_mode=True,
                             optimize_build_volume="pytorch1")

    disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(H, W).clip(0, None)

    _, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    disp[xx - disp < 0] = np.inf

    vis_png = iio.imwrite(
        "<bytes>",
        np.concatenate([orig0, orig1, vis_disparity(disp, color_map=cv2.COLORMAP_TURBO)], axis=1),
        extension=".png",
    )

    depth = cloud = None
    if intrinsics_txt:
        lines    = [l for l in intrinsics_txt.splitlines() if l.strip()]
        K        = np.array(lines[0].split(), dtype=np.float32).reshape(3, 3)
        baseline = float(lines[1])
        K[:2]   *= scale

        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(disp > 0, K[0, 0] * baseline / disp, 0.0).astype(np.float32)

        if o3d:
            xyz  = depth2xyzmap(depth, K)
            pcd  = toOpen3dCloud(xyz.reshape(-1, 3), orig0.reshape(-1, 3))
            z    = np.asarray(pcd.points)[:, 2]
            pcd  = pcd.select_by_index(np.where((z > 0) & (z <= zfar))[0])
            o3d.io.write_point_cloud("/tmp/cloud.ply", pcd)
            cloud = open("/tmp/cloud.ply", "rb").read()

    return {"disp": disp, "vis_png": vis_png, "depth": depth, "cloud": cloud}


def intrinsics_to_txt(path: Path) -> str:
    raw = path.read_text().strip()
    if raw.startswith("{"):
        d = json.loads(raw)
        fx, fy = float(d["fx"]), float(d["fy"])
        cx, cy = float(d["cx"]), float(d["cy"])
        return f"{fx} 0.0 {cx} 0.0 {fy} {cy} 0.0 0.0 1.0\n{float(d['baseline'])}\n"
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left",        required=True)
    ap.add_argument("--right",       required=True)
    ap.add_argument("--intrinsics",  default="")
    ap.add_argument("--out-dir",     default="output")
    ap.add_argument("--weights",     default=str(WEIGHTS_PATH))
    ap.add_argument("--scale",       type=float, default=1.0)
    ap.add_argument("--valid-iters", type=int,   default=8)
    ap.add_argument("--max-disp",    type=int,   default=192)
    ap.add_argument("--zfar",        type=float, default=100.0)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        sys.exit("CUDA not available — this script requires a CUDA GPU")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    weights = Path(args.weights)
    t_load  = time.perf_counter()
    model, _ = load_model(weights)
    print(f"model loaded  {time.perf_counter() - t_load:.2f}s")

    intrinsics_txt = intrinsics_to_txt(Path(args.intrinsics)) if args.intrinsics else None

    t_infer = time.perf_counter()
    r = infer(model,
              Path(args.left), Path(args.right),
              intrinsics_txt,
              scale=args.scale,
              valid_iters=args.valid_iters,
              max_disp=args.max_disp,
              zfar=args.zfar)
    print(f"inference     {time.perf_counter() - t_infer:.2f}s")

    import numpy as np
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "disp_vis.png").write_bytes(r["vis_png"])
    np.save(out / "disp.npy", r["disp"])
    if r["depth"] is not None:
        np.save(out / "depth_meter.npy", r["depth"])
    if r["cloud"] is not None:
        (out / "cloud.ply").write_bytes(r["cloud"])

    print("\n".join(str(p) for p in sorted(out.iterdir())))


if __name__ == "__main__":
    main()
