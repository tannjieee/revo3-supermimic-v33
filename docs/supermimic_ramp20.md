# Ramp20 controlled restart — 2026-10-25

This branch isolates two corrections to Bounded20: restore Geometry-v1's
1000-update actor learning-rate ramp after 500 WM-only iterations, and increase
MJWarp contact capacity using the official public solver configuration.

Production uses 20 Hz control / 120 Hz physics, two GPUs jointly training one
model, 2048 environments per rank, seed 42. All networks, optimizers, statistics,
replay and counters start from zero. The previous run and checkpoints remain in
their original directory. This run does not resume them.

Actor iteration 501 starts at 0.001 times the scheduled learning rate, reaching
the full scheduled rate at iteration 1500. The base rate remains 2e-4. WM and
critic rates and update counts are unchanged. Checkpoint schema remains 12;
the persisted config identifies the ramp and strict config checks reject mixing
different schedules on resume.

Keep tanh actions, Gaussian exploration before tanh, WM k=8, actor H=16, online
critic imagined lambda targets and soft drop continuation unchanged. Near-bound
tanh actions reduce both motor gradients and exploration; the gradual update is
the isolated learning intervention, not proof that all WM issues are solved.
No noise branch, reward, asset, joint order, goal distribution or domain
randomization is changed.

The earlier run emitted broadphase overflow warnings requiring nconmax up to
1192 versus its capacity 1152. Use the public override
`env.sim.physics.solver_cfg.nconmax=1536`; retain njmax=512. Validate at full
environment count with dense action limits and randomization, inspecting MJWarp
overflow, candidate/contact counts and constraint counts. Do not modify vendor
physics code. This capacity change repairs an invalid simulation condition; the
learning comparison is not a strict one-variable experiment because both change.

Run `scripts/supermimic/launch_ramp_scratch.py` with two GPU UUID arguments.
The distinct output is `outputs/revo3_ramp20_2x2048_scratch_seed42`; status is
`outputs/active_ramp20_scratch.json`. Validation results are appended after the
checks finish. Training improvement requires comparable real rollout results;
unit tests and successful launch do not establish that improvement.

## Completed validation and waiting state

- 66 CPU regression tests passed, including two-process synchronization, the
  first policy update, ramp boundaries, restored optimizer learning rates,
  observation/action contracts and saved environment capacity overrides.
- Both RTX 5090 GPUs ran 2048 environments through 8 action phases, 96 steps
  each (768 per rank; 3,145,728 environment transitions). Positive/negative
  limits, mixed limits, reversals, alternating limits and random commands kept
  the task's original resets and randomization. Neither rank emitted capacity
  overflow warnings or nonfinite observations. Peak global collision candidates
  divided by 2048 was 1425.7085, below 1536; peak constraints per world were 345,
  below njmax=512. This finite test does not exhaust every future posture.
- A full-size three-iteration Newton test completed 30 WM, 3 actor and 60 critic
  optimizer updates per rank. Both ranks produced model digest
  `0e34692cd212ffed9df601cb9767386b84ef1a8cbfb89c07377e1e20d05a929f`.
  Network and Adam tensors were finite. Actor rates were 2e-7, 4e-7, 6e-7.
  Reward agreement error was at most 2.98e-8 and FK position error 7.55e-6 m.
  The original profiling wrapper missed the distributed argument; it was fixed
  and the complete test rerun successfully as `evidence/full_joint_check_v2`.
- Actual robot and cube SHA-256 values, environment semantic contract, 20 Hz
  control and physics backend match the previous run. Only contact allocation
  differs in the environment configuration.
- The old run stopped cleanly at iteration 3490; its 192,237,175-byte latest.pt
  and optimizer states remain under the original Bounded20 directory. The last
  500 recorded iterations before stopping (2990–3489) averaged 1.09455 completed
  goals per episode with 66.3404% drops.

Production has **not started**: a new, independent full-hand parameter fitting
job occupied GPU1 immediately before launch. The preflight refused to share it;
the failed preflight logs are preserved. The user explicitly chose to wait for
that fit to finish. `evidence/pending_training.json` records the exact launch
configuration. An active chat follow-up, automation `revo3`, checks every 10
minutes and starts the verified fresh run when GPU1 and GPU3 are available,
then verifies the first checkpoint and stops the follow-up. It must not interrupt
the fit or any other job. This waiting status is historical once that follow-up
has launched training; consult `outputs/active_ramp20_scratch.json` for live state.
