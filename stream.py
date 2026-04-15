# python stream.py [--port 8080] [--scale 0.5]
# Open: http://<jetson-ip>:8080

import argparse
import io
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from camera.oak import (
    init_oak, get_camera_intrinsics,
    build_rectification_maps, create_stereo_queues, rectify_pair,
)
from main import DepthEstimator, WEIGHTS_PATH

_cond = threading.Condition()
_lock = threading.Lock()

# live state
_jpg:         bytes             = b""
_disp:        np.ndarray | None = None
_left_scaled: np.ndarray | None = None  # left BGR at disparity resolution

# camera intrinsics (set once at startup)
_intrinsics: dict | None = None

# captured state (frozen on /capture)
_cap_left_jpg: bytes             = b""
_cap_disp:     np.ndarray | None = None


# ── profile plot ──────────────────────────────────────────────────────────────

def _profile_png(disp: np.ndarray, intrinsics: dict,
                 x0: int, y0: int, x1: int, y1: int) -> bytes:
    import scipy.stats

    fx       = intrinsics["fx"]
    baseline = intrinsics["baseline"]   # metres

    H, W = disp.shape
    n    = max(int(np.hypot(x1 - x0, y1 - y0)) * 2, 2)
    xs   = np.linspace(x0, x1, n).clip(0, W - 1).astype(int)
    ys   = np.linspace(y0, y1, n).clip(0, H - 1).astype(int)
    d    = disp[ys, xs].astype(float)
    dist = np.linspace(0, np.hypot(x1 - x0, y1 - y0), n)

    valid = d > 0
    depth = np.full(n, np.nan)
    depth[valid] = fx * baseline / d[valid]          # metres

    # Floor = belt = farther pixels (higher depth).
    # Use the top 60 % of depth values as floor candidates and fit a
    # linear slope to handle any camera tilt across the line.
    height_mm = np.full(n, np.nan)
    if valid.sum() >= 4:
        lo = np.nanpercentile(depth[valid], 40)
        floor_mask = valid & (depth >= lo)
        if floor_mask.sum() >= 2:
            slope, intercept, *_ = scipy.stats.linregress(
                dist[floor_mask], depth[floor_mask]
            )
            floor = slope * dist + intercept
        else:
            floor = np.full(n, np.nanmedian(depth[valid]))
        # height = how far object is in front of the belt plane, in mm
        height_mm = np.where(valid, np.maximum(floor - depth, 0) * 1000, np.nan)

    fig, ax = plt.subplots(figsize=(8, 3), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(dist, height_mm, color="#1a6fdb", linewidth=1.5)
    ax.fill_between(dist, 0, height_mm, alpha=0.15, color="#1a6fdb",
                    where=~np.isnan(height_mm))
    ax.axhline(0, color="#999", linewidth=0.8, linestyle="--", label="belt")
    ax.set_xlabel("distance along line (px)", color="#333")
    ax.set_ylabel("height above belt (mm)", color="#333")
    ax.tick_params(colors="#333")
    for spine in ax.spines.values():
        spine.set_edgecolor("#ccc")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ── HTML ──────────────────────────────────────────────────────────────────────

_INDEX_HTML = (
    b'<html><body style="margin:0;background:#fff">'
    b'<img src="/stream" style="width:100%;height:100vh;object-fit:contain">'
    b'<div style="position:fixed;top:10px;left:10px">'
    b'<a href="/capture" style="color:#fff;background:#1a6fdb;'
    b'padding:8px 16px;border-radius:6px;text-decoration:none;font-family:sans-serif">'
    b'&#9654; Capture &amp; profile</a></div>'
    b'</body></html>'
)

_INSPECT_HTML = b"""\
<!doctype html><html>
<head><title>Profile Inspector</title>
<style>
  body { margin:0; background:#fff; color:#222; font-family:sans-serif;
         display:flex; flex-direction:column; align-items:center; padding:16px; gap:12px; }
  h2   { margin:0; }
  #wrap { display:flex; gap:20px; align-items:flex-start; flex-wrap:wrap; }
  canvas { cursor:crosshair; border:1px solid #ccc; max-width:640px; }
  #status { color:#666; font-size:.9em; margin-bottom:6px; }
  #plot   { display:none; max-width:640px; border:1px solid #ccc; }
  button  { margin-top:8px; padding:6px 16px; background:#1a6fdb; color:#fff;
             border:none; border-radius:4px; cursor:pointer; }
  button:hover { background:#155bb5; }
</style>
</head>
<body>
<h2>Line Profile</h2>
<div id="wrap">
  <canvas id="c"></canvas>
  <div>
    <div id="status">Click first point</div>
    <img id="plot">
    <br><button id="btn" style="display:none" onclick="reset()">New line</button>
  </div>
</div>
<script>
const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
const base   = new Image();
let pts = [];

base.onload = () => {
  canvas.width  = base.naturalWidth;
  canvas.height = base.naturalHeight;
  ctx.drawImage(base, 0, 0);
};
base.src = '/captured_frame';

function reset() {
  pts = [];
  ctx.drawImage(base, 0, 0);
  document.getElementById('status').textContent = 'Click first point';
  document.getElementById('plot').style.display = 'none';
  document.getElementById('btn').style.display  = 'none';
}

canvas.onclick = e => {
  if (pts.length >= 2) return;
  const r  = canvas.getBoundingClientRect();
  const sx = canvas.width  / r.width;
  const sy = canvas.height / r.height;
  const x  = Math.round((e.clientX - r.left) * sx);
  const y  = Math.round((e.clientY - r.top)  * sy);
  pts.push([x, y]);
  ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
  ctx.fillStyle = pts.length === 1 ? '#00ff80' : '#ff4444';
  ctx.fill();
  if (pts.length === 2) {
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    ctx.lineTo(pts[1][0], pts[1][1]);
    ctx.strokeStyle = '#ffee00'; ctx.lineWidth = 2; ctx.stroke();
    document.getElementById('status').textContent = 'Loading\u2026';
    const pl  = document.getElementById('plot');
    const url = `/profile?x0=${pts[0][0]}&y0=${pts[0][1]}&x1=${pts[1][0]}&y1=${pts[1][1]}`;
    pl.onload = () => {
      document.getElementById('status').textContent = 'Done';
      document.getElementById('btn').style.display = 'inline-block';
    };
    pl.src = url; pl.style.display = 'block';
  } else {
    document.getElementById('status').textContent = 'Click second point';
  }
};
</script>
</body></html>"""


# ── HTTP handler ──────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        p      = parsed.path

        if p == "/":
            self._send(200, "text/html", _INDEX_HTML)

        elif p == "/stream":
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

        elif p == "/capture":
            global _cap_left_jpg, _cap_disp
            with _lock:
                left = _left_scaled
                disp = _disp
            if left is None or disp is None:
                self._send(503, "text/plain", b"no frame yet"); return
            _, enc = cv2.imencode(".jpg", left, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with _lock:
                _cap_left_jpg = enc.tobytes()
                _cap_disp     = disp.copy()
            self.send_response(302)
            self.send_header("Location", "/inspect")
            self.end_headers()

        elif p == "/inspect":
            self._send(200, "text/html", _INSPECT_HTML)

        elif p == "/captured_frame":
            with _lock:
                jpg = _cap_left_jpg
            if not jpg:
                self._send(503, "text/plain", b"no capture yet"); return
            self._send(200, "image/jpeg", jpg)

        elif p == "/profile":
            qs = urllib.parse.parse_qs(parsed.query)
            try:
                x0 = int(qs["x0"][0]); y0 = int(qs["y0"][0])
                x1 = int(qs["x1"][0]); y1 = int(qs["y1"][0])
            except (KeyError, ValueError, IndexError):
                self._send(400, "text/plain", b"bad params"); return
            with _lock:
                disp       = _cap_disp
                intrinsics = _intrinsics
            if disp is None or intrinsics is None:
                self._send(503, "text/plain", b"no capture yet"); return
            self._send(200, "image/png", _profile_png(disp, intrinsics, x0, y0, x1, y1))

        else:
            self.send_response(404); self.end_headers()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import depthai as dai

    ap = argparse.ArgumentParser()
    ap.add_argument("--port",        type=int,   default=8080)
    ap.add_argument("--weights",     default=str(WEIGHTS_PATH))
    ap.add_argument("--scale",       type=float, default=1.0)
    ap.add_argument("--valid-iters", type=int,   default=8)
    ap.add_argument("--max-disp",    type=int,   default=192)
    args = ap.parse_args()

    global _jpg, _disp, _left_scaled, _intrinsics

    est = DepthEstimator(args.weights, args.scale, args.valid_iters, args.max_disp)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"http://0.0.0.0:{args.port}")

    device, calib = init_oak()
    _intrinsics   = get_camera_intrinsics(calib)
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

                    left, right  = rectify_pair(fl, fr, map1_l, map2_l, map1_r, map2_r)
                    vis, disp    = est.infer(left, right)
                    H, W         = disp.shape
                    _, jpg       = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])

                    with _cond:
                        _jpg = jpg.tobytes()
                        _cond.notify_all()
                    with _lock:
                        _disp        = disp
                        _left_scaled = cv2.resize(left, (W, H))

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
