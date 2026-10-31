"""Imaging sonar against the ANALYTIC scene (floor, 4 walls, target cylinder).

The point-cloud version could not show the target because the murky-water map
contains only 2 points on the cylinder. Ray casting against the real geometry --
which is what HoloOcean does with its octree/RTX rays -- gives exact ranges.

Scene (from pool_view.py):
  floor      z = -10.840
  walls      x = +/-12.58, y = -41.0 / +37.2
  cylinder   centre (0,-20), r = 0.25, z in [-12.07, -9.07]  <- the salvage target
  ROV        horizontal imaging sonar, 50 deg sector, 400 beams
"""
import json
import math
import os

import cv2
import numpy as np

M2 = "<REMOTE_HOME>/catkin_ws/holo_ocean/m2_assets"
VIZ = os.path.join(M2, "viz")
OUT = os.path.join(VIZ, "sonar")

FOV_DEG = 50.0
TILT_DEG = -25.0
                        # beam in a pool returns almost nothing (floor is below)
N_BEAMS = 400
RANGE_MAX = 12.0
RANGE_MIN = 0.75
ALPHA_DB = 0.055
NOISE_FLOOR_DB = -72.0
SIGMA_SPECKLE = 0.5

FLOOR_Z = -10.840
WALLS = dict(x_lo=-12.58, x_hi=12.58, y_lo=-41.0, y_hi=37.2)
TGT_C = np.array([0.0, -20.0])
TGT_R = 0.25
TGT_Z = (-12.07, -9.07)


def load_poses():
    vp = json.load(open(os.path.join(VIZ, "vision_poses.json")))["poses"]
    poses = {}
    for a, b in vp.items():
        rv = np.array(b[:3]).reshape(3, 1)
        tv = np.array(b[3:])
        R_f_w = cv2.Rodrigues(rv)[0].T
        poses[int(a)] = (R_f_w, -R_f_w @ tv)
    return poses


def hit_scene(C, d3):
    """First intersection of a ray with the floor plane, the four walls and the
    target cylinder. Returns (range, kind, cos_incidence)."""
    x, y, z = C
    dx, dy, dz = d3
    best = (np.inf, None, 0.4)

    def try_hit(t, kind, cosi):
        if t is None or t <= 0.2 or t >= best[0]:
            return
        px, py, pz = x + t * dx, y + t * dy, z + t * dz
        if not (WALLS["x_lo"] - 1e-6 <= px <= WALLS["x_hi"] + 1e-6
                and WALLS["y_lo"] - 1e-6 <= py <= WALLS["y_hi"] + 1e-6):
            return
        if pz < FLOOR_Z - 1e-6 or pz > 2.4:
            return
        return (t, kind, cosi)

    # floor plane z = FLOOR_Z
    if dz < -1e-6:
        t = (FLOOR_Z - z) / dz
        r = try_hit(t, "floor", 0.55)
        if r:
            best = r
    # walls (vertical planes, any ray that drifts sideways)
    for nx, ny, c0 in ((1, 0, WALLS["x_hi"]), (-1, 0, WALLS["x_lo"]),
                       (0, 1, WALLS["y_hi"]), (0, -1, WALLS["y_lo"])):
        den = dx * nx + dy * ny
        if abs(den) < 1e-9:
            continue
        t = (c0 - (x * nx + y * ny)) / den
        r = try_hit(t, "wall", 0.5)
        if r:
            best = r
    # target cylinder: horizontal circle intersection, then check the z band
    ox, oy = x - TGT_C[0], y - TGT_C[1]
    a = dx * dx + dy * dy
    b = 2.0 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - TGT_R * TGT_R
    disc = b * b - 4.0 * a * c
    if disc > 0 and dz < -1e-6:
        t = (-b - math.sqrt(disc)) / (2.0 * a)
        if t > 0.2:
            pz = z + t * dz
            if TGT_Z[0] <= pz <= TGT_Z[1] and t < best[0]:
                best = (t, "target", 0.92)
    return best


def sonar_image(C, yaw, seed=0):
    ang = np.deg2rad(np.linspace(-FOV_DEG / 2, FOV_DEG / 2, N_BEAMS))
    rng_ = np.random.RandomState(seed)
    bins = np.linspace(RANGE_MIN, RANGE_MAX, 96)
    img = np.full((len(bins), N_BEAMS), NOISE_FLOOR_DB, np.float32)
    for i in range(N_BEAMS):
        a = yaw + ang[i]
        tilt = math.radians(TILT_DEG)
        # horizontal direction x tilt in the vertical plane
        hx, hy = math.sin(a), math.cos(a)
        d3 = np.array([hx * math.cos(tilt), hy * math.cos(tilt), math.sin(tilt)])
        t, kind, cosi = hit_scene(C, d3)
        if kind is None or t > RANGE_MAX:
            continue
        tl = 40.0 * math.log10(max(0.5, t)) + 2.0 * ALPHA_DB * t
        ts = 10.0 * math.log10(max(1e-4, cosi ** 2)) + \
            (14.0 if kind == "target" else 4.0)      # hard target vs concrete
        power = ts - tl
        power += 20.0 * math.log10(max(1e-3, rng_.rayleigh(SIGMA_SPECKLE)))
        k = int(np.searchsorted(bins, t))
        k = min(len(bins) - 1, max(0, k))
        img[k, i] = max(power, NOISE_FLOOR_DB)
        if k + 1 < len(bins):
            img[k + 1, i] = max(img[k + 1, i], power - 9.0)
        # acoustic shadow behind a hard target
        if kind == "target" and k + 4 < len(bins):
            img[k + 4, i] = min(img[k + 4, i], NOISE_FLOOR_DB + 2.0)
    return img, ang, bins


def detect(img, ang, bins):
    db = img - NOISE_FLOOR_DB
    thr = 14.0
    ys, xs = np.nonzero(db > thr)
    best = None
    for y, x in zip(ys, xs):
        if db[y, x] < db[max(0, y - 2):y + 3, max(0, x - 4):x + 5].max():
            continue
        y0, y1 = max(0, y - 2), min(img.shape[0], y + 3)
        x0, x1 = max(0, x - 4), min(img.shape[1], x + 5)
        mwin = db[y0:y1, x0:x1] > thr
        if not mwin.any():
            continue
        _c = mwin.sum(axis=0).nonzero()[0]
        w = int(_c[-1] - _c[0] + 1)
        if w > 20:                    # extended structure -> not a point target
            continue
        # ACOUSTIC SHADOW: a hard target blocks the energy behind it, so the
        # range bins immediately beyond it must be much darker. The seabed arc
        # has echo all the way behind it, which is what tells the two apart --
        # this is the classic imaging-sonar cue and it is what stops the
        # detector from locking onto the floor.
        k = int(y)
        behind = db[min(len(db) - 1, k + 4):min(len(db), k + 9),
                     max(0, x - 3):min(db.shape[1], x + 4)]
        shadow = float(behind.max()) if behind.size else 0.0
        if shadow > 8.0:
            continue
        score = db[y, x] - 2.0 * w + (8.0 - shadow)
        if best is None or score > best[0]:
            best = (score, y, x, w)
    if best is None:
        return None
    _, y, x, w = best
    bearing = float(np.interp(x, np.arange(N_BEAMS), np.degrees(ang)))
    rng = float(np.interp(y, np.arange(len(bins)), bins))
    return dict(bearing_deg=bearing, range_m=rng, peak_db=float(img[y, x]),
                w_beams=w)


def main():
    poses = load_poses()
    os.makedirs(OUT, exist_ok=True)
    print("poses: %d   (analytic scene: floor + 4 walls + target cylinder)"
          % len(poses))
    rows = []
    rows_geo = []
    for f, (R, C) in sorted(poses.items())[::5]:
        fwd = R @ np.array([0, 0, 1.0])
        yaw = math.atan2(fwd[0], fwd[1])
        d_true = float(np.linalg.norm(TGT_C - C[:2]))
        if d_true > RANGE_MAX or d_true < RANGE_MIN:
            continue
        # GEOMETRIC VISIBILITY WINDOW: with a 25 deg downtilt from z=-6, the
        # beam only enters the cylinder's z band (-12.07..-9.07) once the range
        # exceeds (z_cam - z_top)/tan(tilt). Closer than that the beam passes
        # OVER the target -- it is genuinely invisible, not a detection failure.
        import math as _m
        _need = (C[2] - TGT_Z[1]) / _m.tan(_m.radians(abs(TILT_DEG)))
        if d_true < _need:
            rows_geo.append((f, d_true, _need))
            continue
        img, ang, bins = sonar_image(C, yaw, seed=f)
        det = detect(img, ang, bins)
        if det is None:
            continue
        a = yaw + math.radians(det["bearing_deg"])
        est = C[:2] + det["range_m"] * np.array([math.sin(a), math.cos(a)])
        rows.append((f, d_true, det["range_m"],
                     float(np.linalg.norm(est - TGT_C)),
                     det["peak_db"], det["w_beams"]))
    r = np.array(rows, float)
    if rows_geo:
        g = np.array(rows_geo, float)
        print("frames where the target is OUTSIDE the beam's z band "
              "(invisible by geometry): %d  (needs range > %.1f m)"
              % (len(g), np.median(g[:, 2])))
    print("")
    print("frames with the target in the 12 m sector: %d" % len(r))
    print("range error    : median %.3f m  p90 %.3f"
          % (np.median(np.abs(r[:, 2] - r[:, 1])),
             np.percentile(np.abs(r[:, 2] - r[:, 1]), 90)))
    print("position error : median %.3f m  p90 %.3f  <0.5 m %d/%d  <1.0 m %d/%d"
          % (np.median(r[:, 3]), np.percentile(r[:, 3], 90),
             int((r[:, 3] < 0.5).sum()), len(r),
             int((r[:, 3] < 1.0).sum()), len(r)))
    print("echo width     : median %.1f beams" % np.median(r[:, 5]))
    np.save("/tmp/sonar_analytic.npy", r)

    # ---- figure: near + far examples --------------------------------------
    tiles = []
    for want, tag in ((1.5, "near"), (7.5, "far")):
        best = min(((abs(np.linalg.norm(TGT_C - C[:2])) - want, f)
                    for f, (R, C) in sorted(poses.items())), key=lambda t: abs(t[0]))
        f = best[1]
        R, C = poses[f]
        fwd = R @ np.array([0, 0, 1.0])
        yaw = math.atan2(fwd[0], fwd[1])
        img, ang, bins = sonar_image(C, yaw, seed=f)
        det = detect(img, ang, bins)
        d_true = float(np.linalg.norm(TGT_C - C[:2]))
        vis = np.clip((img - NOISE_FLOOR_DB) / 40.0 * 255, 0, 255).astype(np.uint8)
        vis = cv2.applyColorMap(255 - vis, cv2.COLORMAP_BONE)
        vis = cv2.resize(vis, (0, 0), fx=1.7, fy=1.25, interpolation=cv2.INTER_NEAREST)
        if det:
            cx = int((det["bearing_deg"] + FOV_DEG / 2) / FOV_DEG * (vis.shape[1] - 1))
            cy = int((det["range_m"] - RANGE_MIN) / (RANGE_MAX - RANGE_MIN)
                     * (vis.shape[0] - 1))
            cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)
            cv2.circle(vis, (cx, cy), 15, (0, 0, 255), 2)
        cv2.putText(vis, "%s: true %.1f m%s" % (tag, d_true,
                    ("  det %.2f m" % det["range_m"]) if det else "  NO DET"),
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 255), 1)
        tiles.append(vis)
    row = np.zeros((max(t.shape[0] for t in tiles),
                    sum(t.shape[1] for t in tiles), 3), np.uint8)
    x = 0
    for t in tiles:
        row[0:t.shape[0], x:x + t.shape[1]] = t
        x += t.shape[1]
    p = os.path.join(OUT, "analytic_examples.png")
    cv2.imwrite(p, row)
    print("wrote", p)


if __name__ == "__main__":
    main()
