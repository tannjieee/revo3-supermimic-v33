# v30 absolute-position optimization candidate

The Actor emits the same 21 absolute encoder position targets through tanh and the existing affine decoder. No target integrator, rate projector, narrowed joint range, solver change, or weakened physical-health threshold is introduced.

Changes:

- Initialize the final Actor layer at legal encoder HOME, including the thumb CMP offset.
- Use 0.5-degree correlated physical-angle exploration, with replayed noise memory and the same stochastic command transform in imagination.
- Reduce normalized cost weight to 0.05 and Actor learning-rate multiplier to 0.25; add a direct logit penalty outside [-2, 2].
- Fit the bounded six-head Critic from uniform eight-step real windows. End at the first done; failures have no bootstrap, timeouts bootstrap pre-reset physical state. Real behavior returns carry off-policy bias and are not an unbiased current-policy estimator.
- Share a joint-velocity normalization denominator floor across Actor, WM, online and target Critic: 90 degrees/s in physical units, 0.15708 after the existing 0.1 affine scale. Larger empirical standard deviations remain data-driven.
- Record decomposed gradient norms, absolute-target changes due to one Adam update, and real velocity-input normalization diagnostics.

Historical v25-v29 and early v30 checkpoints retain their stored semantics. Missing TD length means one step; missing velocity floor means zero. Resume cannot silently change either setting. Model state and Actor-only exports preserve the normalization buffers.

The user requested further optimization before reviewing formal training. Validation uses finite independent jobs, does not replace `training/current`, and does not start a 1024-environment / 10000-iteration formal run. The old v29 job has already exited on its mechanical-health fail-fast condition at iteration 7578; its final valid saved checkpoint is iteration 7500.

See `validation.json` for actual test, simulation and evaluation evidence, and `../../revo3_network_v30_zh.md` for the complete network, scaling and control contract. Passing implementation checks does not establish reorientation success or hard joint-speed enforcement.
