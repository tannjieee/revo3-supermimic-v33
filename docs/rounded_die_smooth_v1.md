# RoundedDie Smooth-v1 — independent task

Task ID: `Isaac-ViserDex-Repose-Revo3-SuperMimic-RoundedDie-Smooth-v1`.
Parent: the running RSLSphere65 Smooth-v1. This checkout and all outputs are independent.
The user's instruction is to implement and show the loaded environment first. **Do not start training until requested.**
No sphere checkpoint is transferred. Future training initializes from scratch, uses checkpoint version 14,
and rejects a checkpoint whose task ID differs.

## Assets and physics

The supplied archive's `rounded_die` files are copied into `assets/rounded_die` without changing the source
USD, meshes, textures, mass or inertia. It is a roughly 70 mm wooden rounded die, nominal mass 60 g.
Its original convex collider contains 200 vertices and 396 triangles. An independent `object_newton.usda`
layer sets `NewtonMeshCollisionAPI` / `newton:maxHullVertices=200` to avoid Newton's default 64-vertex
convex simplification. The authored principal inertia axes and full tensor are retained.
The original attribution and Gaussian PLY/NPZ (50,000 Gaussians) are included. Simulation preview uses the
original textured mesh. GS is an available appearance asset, not a newly enabled image observation pipeline.

The live adapter validates Newton collision type, unit scale, actual imported vertex positions against the
USD collider, and actual inertia against the mass-randomized authored tensor. The geometry extractor builds
supporting planes from the actual convex hull. Its signed plane margin is exact inside the hull and a lower
bound on Euclidean distance outside edges/corners. It is explicitly a model feature, not a replacement for
Newton's physical collision shape or a claimed exact exterior SDF. Feature dimension remains 300; actor
observations remain 10 × 124. Sphere-only radius and isotropic-inertia assumptions are removed for this task.

## Inherited settings

Implicit PD, base Kp=7 and Kd=0.22, EMA=0.5, 20 Hz control, dt=1/120 with six physics steps and two
Newton substeps, torque limit 0.6 Nm. The saved config's velocity-limit field does not establish an actual
MJWarp hard speed limit. Existing randomization and all sensor-noise/delay/error settings remain unchanged.
The original reset pose/ranges are initially retained and checked in real simulation.

Reward weights: joint velocity −0.00003125, raw action magnitude −0.000125, first action difference
−0.005555555555555556, second action difference −0.005555555555555556. Tracking and success rewards
are inherited. Model/actor/critic settings, 500 WM warmup iterations, 1000 actor ramp iterations,
8-step WM supervision/imagination, UTD=0.4, and BF16 production setup remain unchanged.
Future production uses two idle GPUs, 2048 environments per GPU, total 4096. The launcher refuses occupied
GPUs or existing output directories. No GPU pair is reserved and no production process is launched.

## Preparation and preview

Use the pinned existing Isaac Lab v3.0.0 / Newton 1.6.1 / Warp 1.17.0 runtime without changing dependencies.
`prepare_rounded_die.py` creates a saved environment binding from the parent configuration and verifies
that, apart from object/relocated asset references and the goal marker, the semantic configuration matches.
It never loads a policy or starts simulation/training.

`preview_rounded_die.py` uses the official runtime/launcher and NewtonGL visualizer. It holds the initial
joint targets via the normal action pipeline, checks observations/rewards/EMA/second action differences,
finite geometry gradients, contacts/capacity and timeout reset. The rendered target is visual only,
uses the original mesh/texture, and follows the live goal quaternion at 1:1 scale.
The preview is **not a trained policy demonstration**. Falling or failure to reorient under fixed joint
targets is not a trained-policy success metric.

```bash
bash scripts/run_die.sh scripts/supermimic/preview_rounded_die.py \
  --output_dir outputs/rounded_die_preview --steps 200 --num_envs 1 --viz newton_gl \
  env.sim.physics.solver_cfg.nconmax=2048
```

`start_rounded_die.py` is prepared for a later explicit training request and requires `--start-training`.
Its production configuration is 4096 total environments, scratch initialization, independent logs and weights.
Full two-GPU contact capacity and training throughput have not been validated in this preview-only handoff.

## Validation

CPU regression: 67 passed (convex geometry and gradients, analytic sphere/box regressions, action smoothing,
checkpoint/geometry serialization, legacy version-13 rejection and numerical microbatch equivalence). These are isolated unit fixtures,
not a production training run. Real simulation and protection-check evidence is stored in the sibling
`evidence/` directory and preview output directory; see the completion record appended below.

## Completed preview checks (2026-10-30)

- Single-world seed42: 200 steps / 10 s, fixed initial joint targets, no policy loaded and no optimizer
  updates. The object remained in hand throughout the recorded segment; no automatic reset occurred.
  A separate forced timeout after recording verified the real reset path.
- Live convex vertices: all 200 preserved; maximum imported-coordinate discrepancy 0 m.
  Randomized inertia tensor error 1.09e-11 kg m². Actual sampled mass 56.60 g.
- 64-world seed43: 96 steps/world plus one forced timeout step. Physical parameters and sensor noise
  enabled. 17 automatic resets occurred under fixed joint targets, followed by 64 explicit timeout
  resets. This is an environment check, not a learned-policy evaluation.
- Both checks passed finite-state/geometry checks, observation/reward/EMA/second-difference parity and
  capacity checks. 64-world peaks: 53,830 global collision candidates, 36,011 global contacts and 87
  constraint rows per environment, with configured capacities nconmax1536/njmax512.
- Full video decode: 200 frames, 1280×960, 20 fps, 10 s. SHA-256
  `b08103eb3f965da2054076c75cbe2cc569131839f24bc9425512c7ded1dce6e5`.
  Original mesh and target were visually inspected. NewtonGL used the supported CUDA/GL copy fallback.
- Parent protection: 676 source/config/asset hashes unchanged; original sphere guard and both training
  ranks alive. The new task has no production run directory, checkpoint or running training process.

Remote preview: `outputs/rounded_die_preview_v5/`; diagnostic: `outputs/rounded_die_check64_seed43/`.
The earlier v1-v4 attempts were rejected by integration checks before a valid video was generated;
only v5 is the delivered preview. No engine or installed dependency was modified.

## User-selected lateral placement

The rounded-die initial center is `(0.02, -0.01, 0.565)` m: 10 mm toward
the little-finger side (world -Y), with height and the finger-axis coordinate
unchanged. Reset randomization ranges are unchanged. This placement was
selected for visual review; it is not a claim of improved trained stability.
Training remains unstarted. The sphere task is unaffected.

## Training authorization after preview review

The user accepted the 10 mm lateral placement and requested environment validation,
then a graceful stop and checkpoint preservation of the running sphere Smooth-v1,
followed by this independent task on the same two GPUs. Historical preview-only
status above describes the earlier handoff. Production retains the original
500-iteration world-model warmup and does not load diagnostic or sphere weights.
Actual preflight and launch evidence is in the sibling `evidence/` directory.

## 2026-10-30 RoundedDie production preflight

The accepted lateral reset center is (0.02, -0.01, 0.565) m. Six targeted CPU
regressions and a three-update 64-environment BF16 diagnostic passed, including
forced timeout reset, reward/EMA/second-action-difference parity and actual
convex import checks. The original 200-vertex hull and inertia were preserved.

Two GPUs each passed 768 control steps across eight contact-stress phases with
2048 environments per GPU. Maximum collision candidates were 2,767,412 versus
capacity 3,145,728; maximum contacts 1,904,463; maximum constraint rows 271
versus njmax 512. No overflow or nonfinite observation/reward occurred.

Three full-batch BF16 joint updates passed on the same 4096 environments:
30 WM, 3 actor and 60 critic updates, equal final model hashes across ranks,
and version-14 checkpoint load verified. Last complete iteration was 2.59 s.
Diagnostics used immediate joint updates only to exercise all optimizers.
Production keeps 500 WM warmup iterations and loads no diagnostic weights.
These bounded checks do not establish convergence or all future contact peaks.

Sphere Smooth-v1 stopped gracefully at iteration 6706 (exit 0). Its final
checkpoint, configuration and Git bundle were preserved separately under
`/home/zhangke/rslsphere65_preserved/smooth_stopped_20261030_iter6706`.
Actual launch state and preflight summaries are in the sibling evidence folder.
No engine, installed dependency, sphere source or original asset was modified.
