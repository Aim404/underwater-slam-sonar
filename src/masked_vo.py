"""Minimal masked visual odometry — phase 1 feasibility check.

Deliberately the simplest thing that can work, because the question is only
"does masking let a SLAM front-end run at all":

    ORB -> masked -> ratio-test match -> essential matrix (RANSAC)
        -> recoverPose -> chain -> trajectory

No loop closure, no bundle adjustment, no map. If this drifts, that tells us the
front-end works and the drift is expected; if it fails outright, the masks are the
suspect.

Intrinsics come from pool_view's PinholeCameraCfg (focal 8.0 mm, aperture 12 mm,
640x480), so fx = 640*8/12 = 426.67, fy = fx. Units are metres, camera looking
down its own +Z (OpenGL convention used by Isaac/Blender-style USD cameras).

Ground truth comes from rov_log.csv (px, py, pz), so the output is a real error
number, not just "it ran".

Run on Y:
    ~/anaconda3/envs/isaaclab/bin/python masked_vo.py \
        --seq cam_seq/20261010T153410Z --cam front \
        --mask _manual_mask/manual_mask_front.png \
        --log rov_log/rov_log.csv --step 6
"""
import argparse
import csv
import os

import cv2
import numpy as np


def load_mask(path, shape):
    if not path or not os.path.isfile(path):
        return None
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if (m.shape[1], m.shape[0]) != (shape[1], shape[0]):
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


def load_gt(log_path, dt):
    """step -> (px,py,pz). The recorder names frames <step>.jpg at the physics rate."""
    gt = {}
    with open(log_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                t = float(r["t"])
            except (KeyError, ValueError):
                continue
            gt[int(round(t / dt))] = (float(r["px"]), float(r["py"]), float(r["pz"]))
    return gt


def pose_to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--cam", default="front")
    ap.add_argument("--mask", default="")
    ap.add_argument("--log", default="rov_log/rov_log.csv")
    ap.add_argument("--dt", type=float, default=1.0 / 60.0)
    ap.add_argument("--step", type=int, default=6, help="use every Nth frame")
    ap.add_argument("--max-pairs", type=int, default=400, help="consecutive frame pairs")
    ap.add_argument("--focal", type=float, default=8.0)
    ap.add_argument("--aperture", type=float, default=12.0)
    ap.add_argument("--max-features", type=int, default=3000)
    ap.add_argument("--out", default="_vo")
    args = ap.parse_args()

    d = os.path.join(args.seq, args.cam)
    files = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
    files = files[:: max(1, args.step)][: args.max_pairs + 1]
    if len(files) < 2:
        raise SystemExit("need >=2 frames in %s" % d)
    steps = [int(os.path.splitext(f)[0]) for f in files]

    im0 = cv2.imread(os.path.join(d, files[0]))
    H, W = im0.shape[:2]
    fx = W * args.focal / args.aperture
    fy = fx
    K = np.array([[fx, 0, W / 2.0], [0, fy, H / 2.0], [0, 0, 1.0]])
    mask = load_mask(args.mask, (H, W))
    occ = 100.0 * mask.mean() if mask is not None else 0.0
    print("cam=%s  frames=%d  %dx%d  fx=%.1f" % (args.cam, len(files), W, H, fx))
    print("mask=%s  occluded=%.2f%%" % (args.mask or "(none)", occ))

    gt = load_gt(args.log, args.dt)

    orb = cv2.ORB_create(nfeatures=args.max_features, fastThreshold=7)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    T_wc = np.eye(4)                    # accumulated camera pose in world
    traj = [(steps[0], 0.0, 0.0, 0.0)]
    prev = None
    n_ok = n_fail = n_nomatch = 0
    inl = []

    for k in range(1, len(files)):
        img = cv2.imread(os.path.join(d, files[k]), cv2.IMREAD_GRAYSCALE)
        kp1, des1 = orb.detectAndCompute(im0, mask)
        kp2, des2 = orb.detectAndCompute(img, mask)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            n_nomatch += 1
            im0 = img
            continue
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.8 * n.distance]
        if len(good) < 8:
            n_nomatch += 1
            im0 = img
            continue
        p1 = np.float32([kp1[m.queryIdx].pt for m in good])
        p2 = np.float32([kp2[m.trainIdx].pt for m in good])
        E, inlier = cv2.findEssentialMat(p1, p2, K, method=cv2.USAC_MAGSAC,
                                         prob=0.999, threshold=1.0)
        if E is None or inlier is None or int(inlier.sum()) < 8:
            n_fail += 1
            im0 = img
            continue
        inl.append(int(inlier.sum()))
        _, R, t, _ = cv2.recoverPose(E, p1, p2, K, mask=inlier)
        # k->k+1 motion of the camera; chain it
        T_wc = T_wc @ np.linalg.inv(pose_to_T(R, t))
        traj.append((steps[k], T_wc[0, 3], T_wc[1, 3], T_wc[2, 3]))
        n_ok += 1
        im0 = img

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "vo_%s.csv" % args.cam)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "vo_x", "vo_y", "vo_z", "gt_x", "gt_y", "gt_z", "err"])
        for st, x, y, z in traj:
            g = gt.get(st)
            if g is None:
                w.writerow([st, x, y, z, "", "", "", ""])
            else:
                e = float(np.linalg.norm(np.array([x, y, z]) - np.array(g)))
                w.writerow([st, x, y, z, g[0], g[1], g[2], "%.4f" % e])
    print("\npairs: ok=%d  essential-fail=%d  no-match=%d" % (n_ok, n_fail, n_nomatch))
    if inl:
        print("inliers: median=%d  min=%d  max=%d"
              % (int(np.median(inl)), min(inl), max(inl)))
    print("wrote %s" % csv_path)

    # report against ground truth
    errs = []
    for st, x, y, z in traj:
        g = gt.get(st)
        if g is not None:
            errs.append(float(np.linalg.norm(np.array([x, y, z]) - np.array(g))))
    if errs:
        errs = np.array(errs)
        print("GT error over %d poses: final=%.3f m  median=%.3f  max=%.3f"
              % (len(errs), errs[-1], float(np.median(errs)), errs.max()))
    else:
        print("no GT overlap -- check --step / --dt against the recorder cadence")

    # ---- how much did the vehicle actually move? ----
    common = [(np.array([x, y, z]), np.array(gt[st]))
              for st, x, y, z in traj if st in gt]
    if len(common) > 5:
        V = np.array([p[0] for p in common])
        G = np.array([p[1] for p in common])
        path_v = float(np.linalg.norm(np.diff(V, axis=0), axis=1).sum())
        path_g = float(np.linalg.norm(np.diff(G, axis=0), axis=1).sum())
        print("\ntrue path length = %.3f m over %d poses (%.2f m/frame avg)"
              % (path_g, len(G), path_g / (len(G) - 1)))

        # ---- monocular VO has NO metric scale (recoverPose returns a unit
        # baseline), so a raw metre-to-metre ATE is meaningless. Align with a
        # similarity transform (Sim(3): rotation + translation + one global
        # scale) -- the standard way to score monocular VO.
        def sim3_align(src, dst):
            mu_s, mu_d = src.mean(0), dst.mean(0)
            s0, d0 = src - mu_s, dst - mu_d
            H = s0.T @ d0
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1] *= -1
                R = Vt.T @ U.T
            var = (S ** 2).sum() / max(1e-12, (s0 ** 2).sum())
            c = 1.0 / var
            t = mu_d - c * R @ mu_s
            return c * (R @ src.T).T + t, c, R

        est_al, c, R = sim3_align(V, G)
        ate = np.linalg.norm(est_al - G, axis=1)
        print("Sim(3) aligned: scale=%.4f  rotation=%.2f deg"
              % (c, np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))))
        print("ATE after Sim(3): median=%.3f m  mean=%.3f  final=%.3f  max=%.3f"
              % (float(np.median(ate)), float(ate.mean()), ate[-1], ate.max()))
        print("ATE / path = %.1f%%" % (100 * float(np.median(ate)) / max(1e-9, path_g)))
        np.savetxt(os.path.join(args.out, "traj_%s.csv" % args.cam),
                   np.column_stack([V, G, est_al]),
                   delimiter=",", header="vo_x,vo_y,vo_z,gt_x,gt_y,gt_z,aligned_x,aligned_y,aligned_z",
                   comments="", fmt="%.4f")

    # ---- short-horizon relative error: drift-free by construction, so it shows
    # whether each individual pose estimate is good even if the chain drifts ----
    for H_ in (5, 15):
        errs_h = []
        idx = {st: i for i, (st, *_ ) in enumerate(traj)}
        for st, x, y, z in traj:
            if st not in gt or st + H_ * args.step not in idx:
                continue
            j = idx[st + H_ * args.step]
            if traj[j][0] not in gt:
                continue
            dv = np.linalg.norm(np.array(traj[j][1:4]) - np.array([x, y, z]))
            dg = np.linalg.norm(np.array(gt[traj[j][0]]) - np.array(gt[st]))
            if dg > 1e-6:
                errs_h.append(abs(dv - dg) / dg)
        if errs_h:
            print("RPE over %2d frames: median=%.1f%%  mean=%.1f%%"
                  % (H_, 100 * float(np.median(errs_h)), 100 * float(np.mean(errs_h))))


if __name__ == "__main__":
    main()