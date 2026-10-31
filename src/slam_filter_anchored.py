"""Run the bundle adjustment on the ANCHORED part of a frontend result.

slam_bundle.py refuses the whole problem if ANY camera component lacks 8 unique fixed
marker_xyz points -- correct for a metric map, fatal for a real flight where the vehicle
briefly turns away from the tags (our run loses them around t=19-22 s, giving an
unanchored component of cameras 48-55).

This does not touch their solver. It rebuilds the same bundle_input.npz restricted to the
largest connected component that is properly anchored, then calls slam_bundle.py and
slam_evaluate.py on that subset, and reports how much of the flight survived.

    python3 slam_filter_anchored.py <run_dir> [--min-markers 8]
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

OPT = "<REMOTE_HOME>/anaconda3/envs/tokenizer_bench/bin/python"
HERE = Path("<REMOTE_HOME>/catkin_ws/holo_ocean/m2_assets")


def components(n_cam, obs_camera, obs_point):
    """Union-find over cameras that share a landmark."""
    parent = list(range(n_cam))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_point = {}
    for c, p in zip(obs_camera, obs_point):
        by_point.setdefault(int(p), []).append(int(c))
    for cams in by_point.values():
        for c in cams[1:]:
            union(cams[0], c)
    groups = {}
    for c in range(n_cam):
        groups.setdefault(find(c), []).append(c)
    return sorted(groups.values(), key=len, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--min-markers", type=int, default=8)
    a = ap.parse_args()

    run = Path(a.run).resolve()
    src = run / "bundle_input.npz"
    d = np.load(src, allow_pickle=True)
    K = d["K"]; poses = d["poses"]; points = d["points"]
    oc, op, ouv = d["obs_camera"], d["obs_point"], d["obs_uv"]
    mc, mxyz, muv = d["marker_camera"], d["marker_xyz"], d["marker_uv"]
    fk, tid = d["frame_keys"], d["track_ids"]

    groups = components(len(poses), oc, op)
    print("components (by camera count):", [len(g) for g in groups])

    # markers per camera
    def n_uniq(cam, sel=None):
        m = (mc == cam) if sel is None else (mc[sel] == cam)
        idx = np.where(m)[0]
        if idx.size == 0:
            return 0
        return len(np.unique(np.round(mxyz[idx], 4), axis=0))

    best = None
    for g in groups:
        ok = [c for c in g if n_uniq(c) >= a.min_markers]
        if best is None or len(ok) > len(best):
            best = ok
    if not best or len(best) < 3:
        print("no usable anchored component (best %d cameras)" % len(best or []))
        return 2
    keep = sorted(best)
    print("keeping %d of %d cameras (largest component with >=%d markers each)"
          % (len(keep), len(poses), a.min_markers))
    remap = {c: i for i, c in enumerate(keep)}

    mask_o = np.isin(oc, keep)
    # drop landmarks that no longer have >=2 observations
    op2 = op[mask_o]
    uv2 = ouv[mask_o]
    uniq, counts = np.unique(op2, return_counts=True)
    good = uniq[counts >= 2]
    keep_p = np.isin(op2, good)
    if keep_p.sum() < 15:
        print("too few well-observed landmarks (%d)" % keep_p.sum())
        return 2
    pmap = {p: i for i, p in enumerate(sorted(set(op2[keep_p].tolist())))}

    mask_m = np.isin(mc, keep)
    out = dict(
        K=K,
        poses=poses[keep],
        points=points[sorted(pmap.keys())],
        obs_camera=np.array([remap[int(c)] for c in oc[mask_o][keep_p]], dtype=np.int64),
        obs_point=np.array([pmap[int(p)] for p in op2[keep_p]], dtype=np.int64),
        obs_uv=uv2[keep_p],
        marker_camera=np.array([remap[int(c)] for c in mc[mask_m]], dtype=np.int64),
        marker_xyz=mxyz[mask_m],
        marker_uv=muv[mask_m],
        frame_keys=fk[keep],
        track_ids=tid[sorted(pmap.keys())],
    )
    dst = run / "bundle_input_filtered.npz"
    np.savez(dst, **{k: v for k, v in out.items()})
    print("cameras %d | landmarks %d | obs %d | marker obs %d -> %s"
          % (len(out["poses"]), len(out["points"]), len(out["obs_camera"]),
             len(out["marker_camera"]), dst.name))

    env = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    env = {**__import__("os").environ, **env}
    r = subprocess.run([OPT, "-B", str(HERE / "slam_bundle.py"),
                        "--input", str(dst), "--output", str(run / "bundle_output.npz"),
                        "--max-nfev", "60"], capture_output=True, text=True, env=env)
    print("BA rc=%d" % r.returncode)
    print((r.stdout or "")[-1500:])
    if r.returncode != 0:
        print((r.stderr or "")[-1500:])
        return 3
    r2 = subprocess.run([OPT, "-B", str(HERE / "slam_evaluate.py"), "--run", str(run)],
                        capture_output=True, text=True, env=env)
    print("EVAL rc=%d" % r2.returncode)
    print((r2.stdout or "")[-2000:])
    if r2.returncode != 0:
        print((r2.stderr or "")[-1500:])
        return 4
    ej = run / "evaluation.json"
    if ej.is_file():
        print("\n=== evaluation.json ===")
        print(json.dumps(json.load(open(ej)), indent=1)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
