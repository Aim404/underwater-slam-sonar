#!/usr/bin/env python3
"""Phase-1 visual SLAM feasibility rig (simplest loop that can prove the concept).

  ORB (occlusion mask applied) -> BFMatcher + Lowe ratio
  -> PnP + RANSAC of the current frame against the keyframe's pixels
  -> camera pose + trajectory. New keyframes can be promoted to grow the map.

Driven from the terminal keyboard; the 3D view is published as a JPEG that the
built-in web page renders, so it works over SSH with no GL context.

Keys (no Enter needed -- the loop polls stdin):
  s / SPACE  start / pause
  k          promote the current frame to keyframe (adds 3D points)
  n          step one frame (pauses first)
  r          reset
  c          cycle camera
  w          write points + trajectory to /tmp
  q          quit

Run on Y:
    ~/venvs/slam/bin/python slam_run.py \
        --seq <REMOTE_HOME>/catkin_ws/holo_ocean/m2_assets/cam_seq/<stamp> \
        --mask <REMOTE_HOME>/catkin_ws/holo_ocean/m2_assets/_manual_mask \
        --cam front --port 8780

The sequence folder may hold either layout (auto-detected, JPEGs take priority):
  <stamp>/<cam>/*.jpg   legacy per-frame JPEG folder (pre-2026-10-13 runs)
  <stamp>/<cam>.mp4     one MP4 per camera (the current --record-seq output)
"""
import argparse
import glob
import json
import os
import select
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

# pool_view.py: PinholeCameraCfg(focal_length=8.0, horizontal_aperture=12.0), 640x480
FX = 640.0 * 8.0 / 12.0
CX, CY = 320.0, 240.0
K = np.array([[FX, 0, CX], [0, FX, CY], [0, 0, 1.0]], dtype=np.float64)

CAMS = ["front", "right", "bottom", "rear"]


class FrameSource(object):
    """Frames of one camera from either a legacy JPEG folder or a single MP4.

    The recorder (pool_view.py --record-seq) writes <seq>/<cam>.mp4 since
    2026-10-13; older runs left <seq>/<cam>/*.jpg. Both are supported so the
    2026-10-11 planar-degeneracy baseline stays runnable. stride is applied
    inside, so callers just index 0,1,2,... as before.
    """

    def __init__(self, seq_dir, cam, stride=1):
        self.cam = cam
        self.stride = max(1, int(stride))
        self.kind = None
        self.paths = []
        self.cap = None
        self.n = 0
        self._buf = None
        self._buf_idx = -1
        jpg_dir = os.path.join(seq_dir, cam)
        if os.path.isdir(jpg_dir):
            self.paths = sorted(glob.glob(os.path.join(jpg_dir, "*.jpg")))
            if self.paths:
                self.kind = "jpg"
        if self.kind is None:
            for ext in (".mp4", ".avi", ".mov", ".mkv"):
                p = os.path.join(seq_dir, cam + ext)
                if os.path.isfile(p):
                    cap = cv2.VideoCapture(p)
                    if cap.isOpened():
                        self.kind = "video"
                        self.cap = cap
                        self.path = p
                        self.n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        break
                    cap.release()
        if self.kind == "jpg":
            self.paths = self.paths[::self.stride]
            self.n = len(self.paths)
        elif self.kind == "video":
            # CAP_PROP_FRAME_COUNT can be off by a frame or two; reading past the
            # end simply yields None, which the caller already handles.
            self.n = max(1, self.n_total // self.stride)
        else:
            self.n = 0

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if i < 0 or i >= self.n:
            return None
        if self.kind == "jpg":
            return cv2.imread(self.paths[i])
        want = i * self.stride
        if want < self._buf_idx:              # reset/rewind
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._buf_idx = -1
            self._buf = None
        while self._buf_idx < want:
            ok, fr = self.cap.read()
            if not ok:
                return None
            self._buf_idx += 1
            self._buf = fr
        return self._buf

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>SLAM view</title>
<style>body{background:#111;color:#ddd;font:14px monospace;margin:10px}
img{max-width:100%;border:1px solid #333}
table{border-collapse:collapse;margin-top:8px}
td{padding:2px 14px;border:1px solid #333}.k{color:#ff0}</style></head><body>
<div><span class="k">3D VIEW</span> <span id="st">...</span></div>
<img id="v" src="/view.jpg?t=1">
<table id="tbl"></table>
<script>
async function tick(){
  try{
    const s=await (await fetch('/state?t='+Date.now())).json();
    document.getElementById('st').textContent =
      (s.running?'RUNNING':'PAUSED')+'  cam='+s.cam+'  frame '+s.i+'/'+s.n+
      '  matches='+s.matches+'  inliers='+s.inliers+'  pts='+s.points+'  fps='+s.fps.toFixed(1);
    let h='';for(const kv of Object.entries(s.extra))h+='<tr><td>'+kv[0]+'</td><td>'+kv[1]+'</td></tr>';
    document.getElementById('tbl').innerHTML=h;
    document.getElementById('v').src='/view.jpg?t='+Date.now();
  }catch(e){}
  setTimeout(tick,400);
}
tick();
</script></body></html>"""


class Slam:
    """Two-view bootstrap (essential matrix -> triangulate), then incremental
    PnP of every frame against the growing map, with keyframe promotion to add
    points. This is the minimum that exercises: masked features -> matching ->
    relative pose -> 3D structure -> live 3D view."""

    def __init__(self, cam, args):
        self.cam = cam
        self.args = args
        self.orb = cv2.ORB_create(nfeatures=args.features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.clahe = None if args.no_clahe else cv2.createCLAHE(clipLimit=3.0,
                                                                tileGridSize=(8, 8))
        self.mask = self._mask(cam, args)
        self.reset()

    def _mask(self, cam, args):
        if not args.mask:
            return None
        p = os.path.join(args.mask, "manual_mask_%s.png" % cam)
        if not os.path.isfile(p):
            print("[SLAM] %-7s NO MASK -> whole frame is used" % cam)
            return None
        m = (cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) * 255
        print("[SLAM] %-7s mask %.2f%% occluded (excluded)"
              % (cam, 100 * (m > 0).mean()))
        return 255 - m                       # ORB: 0 = usable, non-zero = ignore

    def reset(self):
        self.P3 = np.zeros((0, 3))       # map points in the current keyframe frame
        self.M3 = np.zeros((0, 3))       # map points in the WORLD frame
        self.desc_map = []               # per-point descriptor rows
        self.kf_pts = np.zeros((0, 2))   # pixels of the current keyframe
        self.desc_kf = None
        self.kf_pose = np.eye(4)         # world -> current keyframe camera
        self.T_w_kf = np.eye(4)         # world -> current KEYFRAME
        self.Trel = np.eye(4)          # keyframe -> current camera
        self.metric = False
        self.kf_id = -1
        self.boot = None                 # (pts0, des0) waiting for its partner
        self.traj = [np.zeros(3)]
        self.inl = 0
        self.nmatch = 0
        self.nfeat = 0
        self.nframes = 0

    def feats(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # CLAHE matters a LOT here: the pool floor is a regular tile pattern, so ORB
        # finds only ~183 keypoints and the essential matrix gets 4 inliers. Equalising
        # the histogram lifts that to ~1500 keypoints / 85 inliers (measured), which
        # is the difference between a degenerate and a solvable two-view bootstrap.
        if self.clahe is not None:
            g = self.clahe.apply(g)
        kp, des = self.orb.detectAndCompute(g, self.mask)
        if des is None or len(kp) < 8:
            return np.zeros((0, 2)), None
        return np.array([k.pt for k in kp], np.float64), des

    def match(self, d1, d2, ratio=None):
        ratio = self.args.ratio if ratio is None else ratio
        out = []
        for pair in self.bf.knnMatch(d1, d2, k=2):
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                out.append((pair[0].queryIdx, pair[0].trainIdx))
        return np.array(out, dtype=np.int32) if out else np.zeros((0, 2), np.int32)

    # ---- bootstrap ---------------------------------------------------------
    def bootstrap(self, pts, des, idx):
        """First frame only: stash it and wait for the next one."""
        self.boot = (pts, des, idx)
        return "boot frame %d stored (%d feats)" % (idx, len(pts))

    def bootstrap_pair(self, pts, des, idx):
        p0, d0, i0 = self.boot
        m = self.match(d0, des)
        self.nmatch = len(m)
        if len(m) < 12:
            self.boot = (pts, des, idx)
            return "bootstrap: only %d matches" % len(m)
        a = p0[m[:, 0]]
        b = pts[m[:, 1]]
        E, emask = cv2.findEssentialMat(a, b, FX, (CX, CY), method=cv2.RANSAC,
                                        prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            self.boot = (pts, des, idx)
            return "bootstrap: E failed"
        n, R, t, pmask = cv2.recoverPose(E, a, b, K, mask=emask)
        self.inl = int(pmask.sum())
        if self.inl < 10:
            self.boot = (pts, des, idx)
            return "bootstrap: only %d inliers" % self.inl
        # triangulate in the frame of image a (= the stashed keyframe)
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = np.hstack([R, t])
        X = cv2.triangulatePoints(P1, P2, a.T, b.T)
        X = (X[:3] / np.where(np.abs(X[3]) < 1e-12, 1e-12, X[3])).T
        good = pmask.ravel().astype(bool) & np.isfinite(X).all(1) & (X[:, 2] > 0.05)
        X = X[good]
        if len(X) < 10:
            self.boot = (pts, des, idx)
            return "bootstrap: only %d triangulated" % len(X)
        # world frame := keyframe frame; the bootstrap keyframe becomes kf 0
        self.P3 = X
        self.M3 = X.copy()
        self.desc_map = [des[m[good][:, 1], :]]
        self.kf_pts = p0
        self.desc_kf = d0
        self.kf_id = i0
        self.kf_pose = np.eye(4)
        self.boot = None
        return "BOOTSTRAP: %d pts from frames %d,%d" % (len(X), i0, idx)

    # ---- tracking ----------------------------------------------------------
    def track(self, pts, des, idx):
        """Estimate the current camera pose.

        Metric path: PnP against the map, used once the map is big enough.
        Fallback: essential matrix against the KEYFRAME (not the map descriptors).
        Matching against the map is hopeless while the map is tiny -- 10 map points
        cannot match against 1500 frame features -- so the bootstrap phase tracks
        against the keyframe's ~1500 descriptors instead and re-anchors.
        """
        if self.args.debug:
            # The 2026-10-15 feasibility run produced "boot frame N stored" over
            # and over with map=0: promote() kept bailing before its debug print,
            # so the state at every frame had to be traced explicitly.
            print("[TRACK] i=%-3d boot=%-5s P3=%-4d kf_pts=%-5d desc_kf=%-5s "
                  "metric=%s"
                  % (idx, "None" if self.boot is None else "f%d" % self.boot[2],
                     len(self.P3), len(self.kf_pts),
                     "-" if self.desc_kf is None else len(self.desc_kf),
                     self.metric), flush=True)
        if len(self.P3) >= 20 and self.desc_map:
            dm = np.concatenate(self.desc_map, 0)
            m = self.match(dm, des)
            if len(m) >= 12:
                obj = self.P3[m[:, 0]].reshape(-1, 1, 3)
                img = pts[m[:, 1]].reshape(-1, 1, 2)
                ok, rvec, tvec, inl = cv2.solvePnPRansac(
                    obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE,
                    iterationsCount=300, reprojectionError=3.0, confidence=0.995)
                if ok and inl is not None and len(inl) >= 10:
                    R, _ = cv2.Rodrigues(rvec)
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = tvec.ravel()
                    self.kf_pose = T                     # world -> camera
                    self.inl = int(len(inl))
                    self.nmatch = len(m)
                    self.traj.append(-R.T.dot(tvec.ravel()))
                    self.metric = True
                    return "PnP inliers %d/%d" % (self.inl, len(m))
            return "map match thin (%d)" % len(m)

        # --- bootstrap phase: essential matrix against the keyframe ---
        # NB: gate on self.boot, not self.kf_pts -- bootstrap() parks the frame in
        # self.boot and leaves kf_pts empty, so testing kf_pts here looped forever
        # on "boot frame stored" and the pair was never formed.
        if self.boot is None:
            return self.bootstrap(pts, des, idx)
        m = self.match(self.desc_kf, des)
        self.nmatch = len(m)
        if len(m) < 12:
            return self.bootstrap_pair(pts, des, idx)
        a = self.kf_pts[m[:, 0]]
        b = pts[m[:, 1]]
        E, em = cv2.findEssentialMat(a, b, FX, (CX, CY), method=cv2.RANSAC,
                                     prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            return self.bootstrap(pts, des, idx)
        n, R, t, pm = cv2.recoverPose(E, a, b, K, mask=em)
        self.inl = int(pm.sum())
        if self.inl < 10:
            return self.bootstrap(pts, des, idx)
        self.Trel = np.eye(4)              # keyframe -> current
        self.Trel[:3, :3] = R
        self.Trel[:3, 3] = t.ravel()
        self.kf_pose = self.T_w_kf @ self.Trel
        self.metric = False
        return "rel-pose inliers %d/%d" % (self.inl, len(m))

    # ---- keyframe promotion: add points by two-view triangulation ----------
    def promote(self, pts, des, idx):
        dbg = self.args.debug

        def _bail(why):
            if dbg:
                print("[PROMOTE] i=%-3d bail: %s" % (idx, why), flush=True)
            return 0

        if len(self.kf_pts) < 8:
            return _bail("kf_pts=%d (<8)" % len(self.kf_pts))
        m = self.match(self.desc_kf, des)
        if len(m) < 12:
            return _bail("keyframe match=%d (<12)" % len(m))
        a = self.kf_pts[m[:, 0]]
        b = pts[m[:, 1]]
        # Re-run the essential-matrix RANSAC here instead of reusing track()'s pose.
        # track() typically keeps only ~25% of the matches (e.g. 33/125); triangulating
        # the 90+ outliers yields points at ~0 depth whose reprojection explodes to
        # 1e5 px, so every point failed the gates and the map stayed empty.
        E, em = cv2.findEssentialMat(a, b, FX, (CX, CY), method=cv2.RANSAC,
                                     prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            return _bail("essential matrix failed")
        _, R, t, pm = cv2.recoverPose(E, a, b, K, mask=em)
        inl = pm.ravel().astype(bool)
        if inl.sum() < 10:
            return _bail("recoverPose inliers=%d (<10)" % int(inl.sum()))
        a, b = a[inl], b[inl]
        Trel = np.eye(4)
        Trel[:3, :3] = R
        Trel[:3, 3] = t.ravel()
        if self.metric and self.P3 is not None and len(self.P3) >= 20:
            P2 = self.kf_pose[:3, :]
        else:
            P2 = Trel[:3, :]
        if P2.shape != (3, 4):
            return 0
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        X = cv2.triangulatePoints(P1, P2, a.T, b.T)
        X = (X[:3] / np.where(np.abs(X[3]) < 1e-12, 1e-12, X[3])).T
        z1 = X[:, 2]
        proj = X @ P2[:, :3].T + P2[:, 3]
        z2 = proj[:, 2]
        safe = np.where(np.abs(z2) < 1e-9, 1e-9, z2)
        uv = np.stack([FX * proj[:, 0] / safe + CX, FX * proj[:, 1] / safe + CY], 1)
        err = np.linalg.norm(uv - b, axis=1)
        good = (z1 > 0.05) & (z2 > 0.05) & np.isfinite(X).all(1) & (err < 4.0)
        if self.args.debug:
            print("[PROMOTE] kf#%d->%d  match=%d  tri_raw=%d  z1>0:%d z2>0:%d "
                  "finite:%d err<4:%d  z1med=%.3f err_med=%.2f"
                  % (self.kf_id, idx, len(m), len(X),
                     int((z1 > 0.05).sum()), int((z2 > 0.05).sum()),
                     int(np.isfinite(X).all(1).sum()), int((err < 4.0).sum()),
                     float(np.median(z1)), float(np.median(err))), flush=True)
        if good.sum() == 0:
            return 0
        Xg = X[good]
        Rt = self.T_w_kf[:3, :3].T
        Cw = -Rt.dot(self.T_w_kf[:3, 3])
        self.P3 = np.vstack([self.P3, Xg]) if self.P3 is not None else Xg
        self.M3 = np.vstack([self.M3, Xg @ Rt.T + Cw]) if self.M3 is not None \
            else Xg @ Rt.T + Cw
        self.desc_map.append(des[m[good][:, 1], :])
        return int(good.sum())

    def reanchor(self, pts, des, idx):
        """Make this frame the keyframe and advance the world->keyframe transform."""
        self.T_w_kf = self.T_w_kf @ self.Trel
        self.kf_pts = pts
        self.desc_kf = des
        self.kf_id = idx
        self.traj.append(self.T_w_kf[:3, 3].copy())

    def world_points(self):
        return self.M3


def render(points, traj, stats, size=(940, 600)):
    img = np.full((size[1], size[0], 3), 20, np.uint8)
    P = np.zeros((0, 3))
    if points is not None and len(points):
        A = np.asarray(points, np.float64)
        A = A[np.isfinite(A).all(1) & (np.abs(A) < 80).all(1)]
        P = A
    T = np.asarray(traj, np.float64) if len(traj) else np.zeros((1, 3))
    if len(P) < 3:
        # still draw the trajectory -- otherwise a run with pose but no map points
        # renders as a completely blank panel (it did, misleadingly)
        lo = T.min(0); hi = T.max(0)
        ctr = (lo + hi) / 2.0
        span = float(max(hi - lo)) * 0.6 + 0.4
        gap = 10
        hw = (size[0] - 3 * gap) // 2
        hh = size[1] - 46
        sc = min(hw, hh) / span

        def tpanel(x0, y0, w, h, a0, a1, ttl):
            u = (T[:, a0] - ctr[a0]) * sc + w / 2
            v = h / 2 - (T[:, a1] - ctr[a1]) * sc
            for k in range(1, len(u)):
                if 0 <= u[k] < w and 0 <= v[k] < h:
                    cv2.line(img, (x0 + int(u[k - 1]), y0 + int(v[k - 1])),
                             (x0 + int(u[k]), y0 + int(v[k])), (0, 230, 120), 2)
            cv2.circle(img, (x0 + int(np.clip(u[0], 0, w - 1)),
                             y0 + int(np.clip(v[0], 0, h - 1))), 5, (0, 255, 255), -1)
            cv2.rectangle(img, (x0, y0), (x0 + w - 1, y0 + h - 1), (70, 70, 70), 1)
            cv2.putText(img, ttl, (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (210, 210, 210), 1)
            cv2.putText(img, "trajectory only - no map points yet",
                        (x0 + 8, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (140, 140, 140), 1)

        tpanel(gap, 32, hw, hh, 0, 1, "top  (x-y)")
        tpanel(2 * gap + hw, 32, hw, hh, 0, 2, "side  (x-z)")
    else:
        lo = np.minimum(P.min(0), T.min(0))
        hi = np.maximum(P.max(0), T.max(0))
        ctr = (lo + hi) / 2.0
        span = float(max(hi - lo)) * 0.6 + 0.4

        # panel geometry first, then derive the scale FROM it -- deriving scale
        # before knowing the panel width is what let points land outside the image.
        gap = 10
        hw = (size[0] - 3 * gap) // 2
        hh = size[1] - 46
        sc = min(hw, hh) / span

        def panel(x0, y0, w, h, a0, a1, ttl):
            q = P[:, [a0, a1]]
            u = (q[:, 0] - ctr[a0]) * sc + w / 2
            v = h / 2 - (q[:, 1] - ctr[a1]) * sc
            ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            img[y0 + v[ok].astype(int), x0 + u[ok].astype(int)] = (255, 190, 90)
            u = (T[:, a0] - ctr[a0]) * sc + w / 2
            v = h / 2 - (T[:, a1] - ctr[a1]) * sc
            for k in range(1, len(u)):
                if 0 <= u[k] < w and 0 <= v[k] < h:
                    cv2.line(img, (x0 + int(u[k - 1]), y0 + int(v[k - 1])),
                             (x0 + int(u[k]), y0 + int(v[k])), (0, 230, 120), 2)
            cv2.circle(img, (x0 + int(np.clip(u[0], 0, w - 1)),
                             y0 + int(np.clip(v[0], 0, h - 1))), 5, (0, 255, 255), -1)
            cv2.rectangle(img, (x0, y0), (x0 + w - 1, y0 + h - 1), (70, 70, 70), 1)
            cv2.putText(img, ttl, (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (210, 210, 210), 1)

        panel(gap, 32, hw, hh, 0, 1, "top  (x-y)")
        panel(2 * gap + hw, 32, hw, hh, 0, 2, "side  (x-z)")
    cv2.putText(img,
                "cam=%s  frame=%d  feats=%d  matches=%d  inliers=%d  pts=%d  traj=%d"
                % (stats["cam"], stats["i"], stats["feats"], stats["matches"],
                   stats["inliers"], 0 if points is None else len(points), len(traj)),
                (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return img


def serve(port, view, sjson):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/":
                d = ("text/html; charset=utf-8", PAGE.encode())
            elif p == "/view.jpg":
                d = ("image/jpeg", view[0] or b"")
            elif p == "/state":
                d = ("application/json", sjson[0])
            else:
                d = ("text/plain", b"?")
                self.send_response(404)
                self.send_header("Content-Length", str(len(d[1])))
                self.end_headers()
                self.wfile.write(d[1])
                return
            self.send_response(200)
            self.send_header("Content-Type", d[0])
            self.send_header("Content-Length", str(len(d[1])))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(d[1])

    srv = HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--mask", default="")
    ap.add_argument("--cam", default="front", choices=CAMS)
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--stride", type=int, default=6,
                    help="Process every Nth recorded frame. 6 (0.3 s of sim time) "
                         "measured best: 66%% of frame pairs triangulate, vs 22%% "
                         "at 20 -- the ROV covers ~0.43 m per sim-second, so a "
                         "1 s stride outruns descriptor matching.")
    ap.add_argument("--features", type=int, default=1500)
    ap.add_argument("--no-clahe", action="store_true",
                    help="disable CLAHE equalisation (it is on by default because the "
                         "tiled pool floor starves ORB without it)")
    ap.add_argument("--debug", action="store_true", help="verbose triangulation stats")
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--autostart", action="store_true")
    ap.add_argument("--headless", action="store_true",
                    help="no tty: run to the end of the sequence")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--kf-every", type=int, default=15,
                    help="auto-promote a keyframe every N processed frames (headless)")
    args = ap.parse_args()

    d = os.path.join(args.seq, args.cam)
    src = FrameSource(args.seq, args.cam, args.stride)
    if len(src) == 0:
        raise SystemExit("no frames in %s (expected %s/*.jpg or %s.mp4)"
                         % (d, d, os.path.join(args.seq, args.cam)))
    print("[SLAM] %s: %d frames (stride %d, %s)  fx=%.1f"
          % (args.cam, len(src), args.stride, src.kind, FX))

    sl = Slam(args.cam, args)
    view, sjson = [None], [b"{}"]
    serve(args.port, view, sjson)
    print("[SLAM] 3D view : http://127.0.0.1:%d" % args.port)
    print("[SLAM] keys    : s=start/pause  n=step  k=add keyframe  r=reset  c=camera  w=save  q=quit\n")

    running = args.autostart
    i = 0
    fps = 0.0
    last = time.time()
    msg = "ready"

    def publish():
        v = render(sl.P3, sl.traj,
                   {"cam": sl.cam, "i": i, "feats": sl.nfeat,
                    "matches": sl.nmatch, "inliers": sl.inl})
        ok, buf = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if ok:
            view[0] = buf.tobytes()
        sjson[0] = json.dumps({
            "running": running, "cam": sl.cam, "i": i, "n": len(src),
            "matches": sl.nmatch, "inliers": sl.inl,
            "points": 0 if sl.P3 is None else int(len(sl.P3)),
            "fps": fps, "traj": [[float(x) for x in t] for t in sl.traj[-300:]],
            "extra": {"msg": msg, "feats": sl.nfeat, "keyframe": sl.kf_id,
                      "mask": "yes" if sl.mask is not None else "NO",
                      "fx": "%.1f" % FX, "cam_dir": d,
                      "pose_t": [round(float(x), 4) for x in sl.kf_pose[:3, 3]]},
        }).encode()

    publish()
    print("[SLAM] ready - press s to run, k to add map points")
    stdin_alive = sys.stdin is not None and sys.stdin.isatty()
    if not stdin_alive:
        print("[SLAM] no interactive tty -> running headless (s starts, k auto every 25 frames)")
    while True:
        ch = ""
        if stdin_alive:
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0)
                if r:
                    line = sys.stdin.readline()
                    if line == "":
                        stdin_alive = False
                        print("[SLAM] stdin closed -> headless")
                    else:
                        ch = line.strip().lower()
            except Exception:
                stdin_alive = False
        elif args.headless:
            # headless: no keys, so just run (bounded by --max-frames)
            # NB: the sequence ending must ALSO break out. This branch used to set
            # running=True unconditionally, so once i hit len(src) the loop span
            # forever on "end of sequence" (621k lines in one run) unless
            # --max-frames happened to exceed the frame count.
            if i >= len(src):
                print("[SLAM] headless run complete: %d frames processed, "
                      "map=%s" % (i, "empty" if sl.P3 is None or len(sl.P3) == 0
                                  else "%d pts" % len(sl.P3)), flush=True)
                break
            if args.max_frames and i >= args.max_frames:
                break
            running = True

        # headless: promote a keyframe every N frames so the map can grow
        if not stdin_alive and running and i % max(5, args.kf_every) == 0:
            ch = "k"

        step = False
        if ch == "q":
            print("[SLAM] quit")
            break
        elif ch in ("s", " "):
            running = not running
            print("[SLAM] %s" % ("RUNNING" if running else "PAUSED"))
        elif ch == "n":
            running = False
            step = True
        elif ch == "r":
            sl.reset()
            i = 0
            print("[SLAM] reset")
        elif ch == "c":
            nxt = CAMS[(CAMS.index(sl.cam) + 1) % 4]
            nxt_src = FrameSource(args.seq, nxt, args.stride)
            if len(nxt_src) > 0:
                sl = Slam(nxt, args)
                src = nxt_src
                d = os.path.join(args.seq, nxt)
                i = 0
                print("[SLAM] camera -> %s (%d frames, %s)"
                      % (nxt, len(src), src.kind))
            else:
                print("[SLAM] no frames for %s" % nxt)
        elif ch == "w":
            np.save("/tmp/slam_points_%s.npy" % sl.cam,
                    np.zeros((0, 3)) if sl.P3 is None else sl.P3)
            np.save("/tmp/slam_traj_%s.npy" % sl.cam, np.array(sl.traj))
            print("[SLAM] saved /tmp/slam_{points,traj}_%s.npy" % sl.cam)

        if i < len(src) and (running or step):
            img = src[i]
            if img is None:
                msg = "read failed"
            else:
                pts, des = sl.feats(img)
                sl.nfeat = 0 if pts is None else len(pts)
                if des is None:
                    msg = "no features"
                else:
                    if ch == "k":
                        # track() MUST run first: promote() triangulates against the
                        # keyframe->current transform (self.Trel) that track() computes.
                        # Promoting without it silently used a stale pose and always
                        # returned 0 points, so the map never grew.
                        msg = sl.track(pts, des, i)
                        n = sl.promote(pts, des, i)
                        sl.reanchor(pts, des, i)
                        msg = "kf#%d %s | +%d pts | map=%d" % (i, msg, n, len(sl.P3))
                        print("[SLAM] %s" % msg)
                    else:
                        msg = sl.track(pts, des, i)
            i += 1
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(1e-3, now - last)
            last = now
            if i % 10 == 0:
                print("[SLAM] f%-4d %-26s feats=%-5d inl=%-5d pts=%-6d"
                      % (i, msg, sl.nfeat, sl.inl, 0 if sl.P3 is None else len(sl.P3)))
        elif i >= len(src):
            if running:
                print("[SLAM] end of sequence (%d frames)" % len(src))
            running = False
            if not stdin_alive and args.max_frames and i >= args.max_frames:
                break
            if stdin_alive:
                break          # interactive: park at the end, user can still reset
        publish()


if __name__ == "__main__":
    main()