# Underwater SLAM and imaging sonar — a simulation study

What a robot can still know about where it is, when the water takes away most of what vision gives.

This repository holds the perception study behind my master's work on a deep-water salvage ROV: a
pool-scale scene in Isaac Sim with a **turbidity parameter I can sweep**, a **visual SLAM chain**, and
a **forward-looking imaging sonar implemented from its physics**. The point of the study is not a new
SLAM algorithm. It is to measure **where each modality stops working**, and what that boundary looks
like as a function of visibility.

**All results are from simulation.** The sonar is synthesised from its physics, not measured by a real
device. This is stated up front because it changes what the numbers mean.

---

## Key numbers

Two recorded sequences, same scene, same pipeline:

| Sequence | Water | Localisable frames | Median error | RMSE | 0.3 s RPE RMSE |
|---|---|---:|---:|---:|---:|
| 2026-10-18 (136 s) | clear, ~31 m visibility | 268/454 → **441/454 (97.1 %)** | 5.95 → **3.71 cm** | 7.22 → 4.39 | 6.89 → 2.06 |
| 2026-10-19 (112 s) | turbid, ~8 m visibility | 330/374 → **349/374 (93.3 %)** | 7.15 → **4.13 cm** | 9.47 → 5.58 | 10.96 → 5.01 |

All errors are measured **in the AprilTag world frame, without ground-truth alignment and without
scale fitting** — the tag map defines the scale, and ground truth is used only for scoring.
An independent recomputation (all-frame median) gives **0.0448 / 0.0422 m**, consistent with the above.

Imaging sonar, inside the geometric visibility window:

| Metric | Result |
|---|---|
| Target range error (median) | **0.248 m** |
| Target position error | **0.295 m** |
| Trials under 0.5 m | 4 / 4 |
| Mapping coverage (112 s sequence) | 1800 cells / 5.7 % |

**Complementarity — the result I care about most:** vision resolves **0.09 m within 8 m**; the sonar
holds **0.30 m out to 12 m**. They fail in different regimes, which is why the interesting question is
not how to merge them but where to hand over.

---

## Interactive demo

[`demo/slam_3d.html`](demo/slam_3d.html) — open it in a browser (no server, no dependencies; it is a
single self-contained file). It shows the estimated trajectory against ground truth and the camera
frustums, scrubbable in time.

---

## What is in here

```
src/slam_run.py                 visual SLAM driver (tag anchoring, KLT front end, triangulation,
                                PnP relocalisation, joint sparse bundle adjustment)
src/masked_vo.py                mask-constrained visual odometry core
src/slam_filter_anchored.py     anchored pose filtering and drift correction
src/sonar_analytic.py           forward-looking imaging sonar — analytic model
figures/                        evaluation plots, sonar panels, elevation map, underwater render
demo/slam_3d.html               interactive trajectory + frustums view
```

## How the sonar is built

No part of this is a wrapper around an existing sonar simulator. The forward-looking sonar is
implemented from the acoustic model:

- range and azimuth sampling from a **ray-cast** against the scene geometry
- target strength `TS = 10 log10(cos²θ) + TS_material` (target 14 dB, board 4 dB, concrete 4 dB)
- transmission loss `TL = 40 log10(r) + 2αr` with `α = 0.055 dB/m` at ~300 kHz
- **Rayleigh (Swerling 1) speckle** — without it the imagery does not look like sonar
- **acoustic shadow** behind hard targets, and a configurable vertical beam wedge (not a flat fan)
- pose-accumulated mapping for the elevation map

The sensor is mounted where the vehicle model puts it, read from the ROV's own `sonar_link`.

## How the scene is built

Isaac Sim 4.5 / Isaac Lab. Pool 25 × 78 m, ROV with four synchronised cameras (front / rear / right /
bottom), six AprilTag boards (0.28 m squares), a cylindrical target (r = 0.25 m, h = 3 m), and a
turbidity model with a live parameter dashboard: visibility, scattering, fog, marine snow, caustics,
attenuation mode — plus a one-click reset that restores the initial pose and clears the sonar buffer.

Marine snow is **2600 instanced spheres** and caustics are **emissive geometry on the floor**, not
post-process overlays, so particles occlude and caustics respect surface orientation. Early versions
used full-frame overlays; they were replaced because an overlay has no physical relationship to the
scene.

An earlier version of this pipeline ran on **ROS–Gazebo with UUV Simulator** (with domain
randomisation and a teleoperation GUI); the scene was later ported to Isaac Sim.

---

## Honest scope

These are the limitations, stated plainly:

1. **Everything is simulated.** No real vehicle, no real sonar. The numbers speak to the algorithm,
   not to a field deployment.
2. **The two chains are not coupled.** They share recorded sequences and not an optimiser. What is
   measured is where each modality stops working — not a fused estimator. Online tight coupling is the
   obvious next step and it is not done here.
3. **The visual chain is offline**, with two-way initialisation. There is no online keyframe
   management, no non-adjacent loop closure, and no marker-free mapping.
4. **The two sequences are not a controlled experiment.** Turbidity differs, but so does the
   trajectory; the comparison is suggestive rather than causal.
5. **Sonar mapping coverage is 5.7 %**, and that is geometry, not a defect: the vehicle cruises at
   4.8 m altitude and a −25° beam grazes the top of a 3 m target.
6. **The sonar detector confuses walls with targets** — walls cast shadow too. A curvature-based fix is
   designed but not implemented.
7. Paths beginning with `<REMOTE_HOME>` are from the development machine and need to be pointed at
   your own copy of the asset tree.

## Not included

- the recorded image sequences and sonar buffers (size)
- the Isaac Sim / Isaac Lab scene construction scripts (they depend on a specific asset tree)
- a separate project on underactuated gripper dynamics, which is under journal review and covered by
  patents

## Contact

Yuncong Yang — Underwater Engineering Institute, Shanghai Jiao Tong University
[aim404@sjtu.edu.cn](mailto:aim404@sjtu.edu.cn) · [aim404.github.io](https://aim404.github.io)
