# python main.py --left L.png --right R.png [--intrinsics K.json]

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

FFS_PATH     = Path(__file__).parent / "external" / "Fast-FoundationStereo"
WEIGHTS_PATH = FFS_PATH / "weights" / "23-36-37"


def _ensure_ffs():
    if str(FFS_PATH) not in sys.path:
        sys.path.insert(0, str(FFS_PATH))


def load_model(weights: Path):
    import torch, yaml
    _ensure_ffs()
    ckpt = weights / "model_best_bp2_serialize.pth"
    if not ckpt.exists():
        sys.exit(f"weights not found: {ckpt}")
    with open(weights / "cfg.yaml") as f:
        yaml.safe_load(f)
    model = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    model.cuda().eval()
    return model


class DepthEstimator:
    """Load FFS once; call infer() per frame pair."""

    def __init__(self, weights=WEIGHTS_PATH, scale=1.0, valid_iters=8, max_disp=192):
        import torch
        _ensure_ffs()
        from core.utils.utils import InputPadder
        try:
            from Utils import AMP_DTYPE
        except ImportError:
            AMP_DTYPE = torch.float16

        if not torch.cuda.is_available():
            sys.exit("CUDA not available")

        self.scale, self.valid_iters, self.max_disp = scale, valid_iters, max_disp
        self._Padder    = InputPadder
        self._amp_dtype = AMP_DTYPE
        self.model      = load_model(Path(weights))
        self.model.args.valid_iters = valid_iters
        self.model.args.max_disp    = max_disp
        torch.autograd.set_grad_enabled(False)

    def infer(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        """Returns side-by-side BGR: [left | turbo depth]. Close = red, far = blue."""
        import torch

        img0 = cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

        if self.scale != 1.0:
            img0 = cv2.resize(img0, None, fx=self.scale, fy=self.scale)
            img1 = cv2.resize(img1, (img0.shape[1], img0.shape[0]))

        H, W = img0.shape[:2]
        t0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
        t1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)

        padder = self._Padder(t0.shape, divis_by=32, force_square=False)
        t0, t1 = padder.pad(t0, t1)

        with torch.amp.autocast("cuda", enabled=True, dtype=self._amp_dtype):
            disp = self.model.forward(t0, t1, iters=self.valid_iters,
                                      test_mode=True, optimize_build_volume="pytorch1")

        disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(H, W).clip(0, None)

        _, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        disp[xx - disp < 0] = 0

        valid = disp > 0
        norm  = np.zeros((H, W), dtype=np.uint8)
        if valid.any():
            lo, hi = disp[valid].min(), disp[valid].max()
            if hi > lo:
                norm[valid] = ((disp[valid] - lo) / (hi - lo) * 255).astype(np.uint8)

        return np.concatenate([cv2.resize(left_bgr, (W, H)),
                                cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)], axis=1)

    def infer_full(self, left_bgr, right_bgr, intrinsics_txt=None, zfar=100.0):
        import imageio.v3 as iio, torch
        _ensure_ffs()
        from Utils import vis_disparity, depth2xyzmap, toOpen3dCloud, o3d

        img0 = cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
        if self.scale != 1.0:
            img0 = cv2.resize(img0, None, fx=self.scale, fy=self.scale)
            img1 = cv2.resize(img1, (img0.shape[1], img0.shape[0]))

        H, W  = img0.shape[:2]
        orig0 = img0.copy()

        t0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
        t1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
        padder = self._Padder(t0.shape, divis_by=32, force_square=False)
        t0, t1 = padder.pad(t0, t1)

        with torch.amp.autocast("cuda", enabled=True, dtype=self._amp_dtype):
            disp = self.model.forward(t0, t1, iters=self.valid_iters,
                                      test_mode=True, optimize_build_volume="pytorch1")

        disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(H, W).clip(0, None)
        _, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        disp[xx - disp < 0] = np.inf

        vis_png = iio.imwrite("<bytes>",
            np.concatenate([img0, cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB),
                            vis_disparity(disp, color_map=cv2.COLORMAP_TURBO)], axis=1),
            extension=".png")

        depth = cloud = None
        if intrinsics_txt:
            lines    = [l for l in intrinsics_txt.splitlines() if l.strip()]
            K        = np.array(lines[0].split(), dtype=np.float32).reshape(3, 3)
            baseline = float(lines[1])
            K[:2]   *= self.scale
            with np.errstate(divide="ignore", invalid="ignore"):
                depth = np.where(disp > 0, K[0, 0] * baseline / disp, 0.0).astype(np.float32)
            if o3d:
                xyz = depth2xyzmap(depth, K)
                pcd = toOpen3dCloud(xyz.reshape(-1, 3), orig0.reshape(-1, 3))
                z   = np.asarray(pcd.points)[:, 2]
                pcd = pcd.select_by_index(np.where((z > 0) & (z <= zfar))[0])
                o3d.io.write_point_cloud("/tmp/cloud.ply", pcd)
                cloud = open("/tmp/cloud.ply", "rb").read()

        return {"disp": disp, "vis_png": vis_png, "depth": depth, "cloud": cloud}


def _intrinsics_to_txt(path: Path) -> str:
    raw = path.read_text().strip()
    if raw.startswith("{"):
        d = json.loads(raw)
        return (f"{d['fx']} 0.0 {d['cx']} 0.0 {d['fy']} {d['cy']} 0.0 0.0 1.0\n"
                f"{d['baseline']}\n")
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

    t0  = time.perf_counter()
    est = DepthEstimator(args.weights, args.scale, args.valid_iters, args.max_disp)
    print(f"loaded {time.perf_counter()-t0:.2f}s")

    left_bgr  = cv2.imread(args.left)
    right_bgr = cv2.imread(args.right)
    if left_bgr is None:  sys.exit(f"cannot read {args.left}")
    if right_bgr is None: sys.exit(f"cannot read {args.right}")

    intrinsics_txt = _intrinsics_to_txt(Path(args.intrinsics)) if args.intrinsics else None

    t0 = time.perf_counter()
    r  = est.infer_full(left_bgr, right_bgr, intrinsics_txt, args.zfar)
    print(f"infer  {time.perf_counter()-t0:.2f}s")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "disp_vis.png").write_bytes(r["vis_png"])
    np.save(out / "disp.npy", r["disp"])
    if r["depth"] is not None: np.save(out / "depth_meter.npy", r["depth"])
    if r["cloud"] is not None: (out / "cloud.ply").write_bytes(r["cloud"])
    print("\n".join(str(p) for p in sorted(out.iterdir())))


if __name__ == "__main__":
    main()
