# Revo3 geometry world model v1

Task: `Isaac-ViserDex-Repose-Revo3-SuperMimic-Geometry-v1`.
Branch: `codex/revo3-geometry-wm`. Built from the independent EventRamp version.

## Model and training contract

The physical output remains 54 tangent-coordinate increments: joint position21,
joint velocity21, object position3, rotation3, linear velocity3, angular velocity3.
Physical state55, full state172, randomized context548, actor124×10 and action21
retain their meanings. One shared model is trained synchronously by two ranks,
2048 environments per rank, total4096, 30Hz. New production training starts from
scratch; no teacher, old actor, or diagnostic checkpoint is loaded.

The active compact-v4 hand and exact-box RSLCube assets are read without edits.
`geometry_asset.py` reads USD Physics joint body0/body1 frames, axes and actual
collision vertices. It orders the tree independently of USD traversal, using the
policy joint names. It extracts 22 bodies and 30 landmarks: 21 collision-center
representatives (joint-origin fallback only for a link without collision), five
distal representatives and four actual grasp-facing palm surface vertices.
The current active asset needs no fallback. These are geometric landmarks,
not measured contact locations. The root-authored .9 scale is included. Box half extents are approximately .02925m;
the exact transformed values are stored in geometry.json. Extraction explicitly
supports the current single, body-aligned Cube collision primitive and rejects
other object shapes rather than silently substituting a bounding box.

`geometry.py` implements batched differentiable FK over five tree levels:
Tchild = Tparent Xparent Raxis(q) inverse(Xchild). It includes nonzero child-frame
offsets, link angular velocities and the velocity of each offset landmark.
For each landmark, the 10 extra features are object-frame relative position3,
relative surface velocity3, point-to-box signed distance1 and outward normal3.
Relative velocity subtracts object translation AND omega×radius. This adds300
WM/Critic inputs (926/1024 total); actor inputs are unchanged. Every imagined
step recomputes geometry from its predicted physical state. No real future
geometry enters imagination, and no independently drifting geometry state exists.
The box distance is an analytic feature; it does not change the Newton collider.

Features are preconditioned by .05m/.5m/s/.02m before RunningNorm. Normalizers
still use real data only. Geometry and integration run FP32; only MLP matmuls
use BF16. The optional compiler fuses the same geometry formulas. It does not
introduce CUDA graphs or alter the simulator. `--eager_geometry` selects the
unfused implementation for diagnostics.

## Objective

Each of six normalized physical residual groups is averaged internally, then
the six group means receive equal weight. This replaces the implicit42:12
hand/object weighting from averaging all54 coordinates. Existing per-coordinate
normalization and floors are retained.

Additional geometric supervision has five group means: hand landmark position,
hand landmark velocity, object-relative position, relative velocity, and surface
distance. SI error scales are .01m/.1m/s/.01m/.1m/s/.01m. Their mean is multiplied
by0.5, then added to the physical objective. These scales and weights are this
Revo3 adaptation's explicit initial choices, not upstream paper defaults.
Normals are inputs only; no discontinuous edge-normal target is fitted.

Targets are reconstructed from recorded real pre-reset states through the same
fixed FK, whose positions, quaternions and velocities are independently checked
against official live articulation link data. This avoids expanding the replay
with redundant geometry. It retains8-step autoregressive WM training,32-step
imagination,10 WM/1 actor/20 critic updates,500 WM-only iterations,1000-update
actor ramp and the existing25% EventRamp archive. Rewards, control mapping,
task termination, sensor model, domain randomization and physics are unchanged.
There is no force/impulse prediction head; no contact-force labels are collected.

## Checkpoints and execution

Geometry checkpoints use version4 and include the complete geometry description,
configuration, model/normalizer/optimizer states and RNG states. Restore checks
geometry as well as the existing environment and observation contract. Direct
resume from a geometry-free model is not supported. Actor-only FP32 TorchScript
export remains `[batch,10,124] -> [batch,21]`, without privileged inputs.

Run through `scripts/run_newton.sh`, which supplies the official pinned runtime
and the environment's existing compiler headers. Bare Python does not supply
those headers to Triton on this server.

## Validation

- 36 distinct contract tests pass in the official runtime (the35-test suite,
  followed by the updated11-test geometry suite including the new scale test), including independent NumPy
  homogeneous transforms for the actual22-body tree, analytical and finite-
  difference velocities, geometry derivatives, microbatch equivalence, checkpoint
  identity, actor gradients through frozen WM and two-process synchronization.
- 64-env real Newton smoke passed reset, reward, observation and live FK checks.
- Unfused2048×2 full-batch diagnostic completed3 joint iterations: each rank
  WM30/Actor3/Critic60; final model digests agree. Max FK errors across ranks:
  position7.62e-6m, linear velocity6.70e-5m/s, angular velocity2.29e-5rad/s.
  Full network update after startup:3.88–3.89s. Fresh32-step holdouts are saved.
- The root-scale regression checks that a .065m primitive under a .9-scaled
  rigid root produces .0585m physical edge length. The final live Newton
  shape_scale comparison reports zero half-extent error.
- Final compiled CUDA parity passed: feature max absolute difference7.63e-6,
  feature-gradient difference1.23e-4, auxiliary loss difference1.23e-6,
  loss-gradient difference9.54e-7. Parameter/buffer checkpoint keys are unchanged.
- Final compiled2048×2 diagnostic completed3 joint iterations and32 held-out
  steps per rank. Per-rank counters WM30/Actor3/Critic60; matching model digest
  `42431f4c4829f14d0bd8c8eaa048f0c3f8f5eb5a6c97f7714cc8f4fc5e9c8e17`.
  Network update durations were184.68s (first compilation),3.40s,1.79s.
  These are short-run measurements, not a long-run throughput estimate.
  Reward parity max2.98e-8; FK position max7.63e-6m across ranks.
- Actual version4 full-size checkpoint loaded successfully; actor export and
  reload parity passed for batch sizes1 and7. Diagnostic checkpoint weights
  are not used in production. Tests do not establish training convergence.

## Production launch

Server22021, GPUs0/4 (UUIDs ending b176dbb / c3316811), one synchronous model.
Root: `/home/zhangke/viserDex_supermimic_geometry_20261024`.
Run: `outputs/revo3_geometry_2x2048_scratch_seed42`.
State: `outputs/active_geometry.json`.
Code revision on server: `2c871a3c0c99e8717709c31a0aceb4d75be52619`.
All21 source/test files in the local manifest match the deployed files.

EventRamp was stopped gracefully at iteration6461 with exit0 and its final
checkpoint retained. Baseline GPU1/5 training PIDs175860/175861 were still
running after the new launch. This experiment preserves those baseline files.

Launch uses the existing process-status runner and official pinned runtime:

```bash
export CUDA_VISIBLE_DEVICES=GPU-f5aba079-86bd-2df0-cac5-68663b176dbb,GPU-ac561d84-ce6e-f1ac-82ee-8c9dc3316811
export UV_BIN=/home/zhangke/isaaclab-ga-upgrade-20261020/bin/uv
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2
bash scripts/run_newton.sh -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/supermimic/train.py --distributed --num_envs 2048 --viz none --seed 42 \
  --env_config /home/zhangke/viserDex_revo3/outputs/student_dagger_start_20261022/frozen_teacher/params/env.yaml \
  --output_dir outputs/revo3_geometry_2x2048_scratch_seed42 \
  --iterations 320000 --save_interval 2000 --latest_interval 100 --amp bf16 \
  --wm_microbatch 2048 --actor_microbatch 4096 --critic_microbatch 32768 --holdout
```

The production command has no resume, smoke, forced-reset or immediate-actor
diagnostic flags. It retains500 WM-only iterations and the1000-update actor ramp.

Production confirmed through iteration121 in the WM-only phase. PIDs269561/269562
are both active; the iteration100 version4 checkpoint (147449351 bytes) loaded
successfully with WM1000/Actor0/Critic0,30 landmarks and no parent checkpoint.
Both ranks passed the iteration100 parameter/normalizer equality check, digest
`554be583a43a7a6894bf8afab8887a6ec44cc7ec8dd8b4b7706b190342c9a422`.
At this check each GPU used15329MiB; the last20 WM update durations had median1.04s.
These are early warmup measurements, not joint-training memory or task success.

## Sources

G1 uses analytical differentiable FK in its reward/Critic path:
https://github.com/zita-ch/supermimic_vibed/blob/a1508d42c9d1699a52e8233ab16d08c488441ab9/supermimic/kinematics.py

Go2 directly predicts hip/knee/foot coordinates and uses them in geometric rewards:
https://github.com/Zhengtianjin/go2-supermimic/blob/054a13c13285a1c94b2e69dc8ac7331f9a8c71cf/go2_supermimic/supermimic/state.py

The combined geometry-input/supervision design here is an adaptation, not an
unchanged reproduction of either repository.
