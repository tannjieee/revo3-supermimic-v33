# Bounded20 correction, 2026-10-25

Current branch: `codex/supermimic-bounded20`; task `Isaac-ViserDex-Repose-Revo3-SuperMimic-Bounded20-v1`; checkpoint schema 12. The earlier Geometry20 notes below are historical, not the current configuration.

The frozen schema-11 iteration-4800 audit found 43.96% deterministic action components outside [-1,1]. The official joint-limit transform clamps these commands, cutting the motor-path derivative. Actor gradients in two sampled batches were dominated by the terminal value bootstrap. Matched Newton perturbations also showed weak object-rotation response: roughly 27–30% of one-step directional responses had a negative dot product with the real response, and predicted rotation response amplitude was about 25–30% of the measured RMS. These are evidence of problems, not proof that one intervention explains the regression. Long contact rollouts also accumulate simulator repeatability noise.

Implemented changes:

- Actor output is FP32 `tanh(logits)`. Exploration is `tanh(logits + 0.1 * Gaussian)`. Real replay, imagined history, evaluation and export use this same bounded command. The official joint-limit mapping and EMA remain unchanged. This is a Revo3 action-interface adaptation; the reference robot's action parameterization is not copied blindly. Tanh can still approach saturation, so log both `abs(action)>0.98` and its mean derivative `1-action²`, not just hard clipping.
- Critic learns exclusively from imagined lambda returns, using the same pre-update online value network as the actor. Targets are detached once, before all 5 x 4 critic updates. Remove the extra EMA critic and 50% real TD loss. This matches the cited implementations' critic objective. Input normalizers still use real data only, so this is not a claim of identical implementation in every detail.
- First 500 iterations fit only the WM; real observation statistics accumulate during this phase. Actor and critic weights/Adam states are untouched until joint training. Actor LR returns to the shared cosine schedule with nominal 2e-4 and no extra 0.25 multiplier or 1000-iteration ramp.
- Keep 8-step autoregressive WM fitting and all six physical / five geometry supervision groups. Use 16 imagined control steps, or 0.8 seconds at 20 Hz. This is an engineering choice between the previous 8-step bootstrap-heavy rollout and 32 steps of larger model drift, not an asserted optimal horizon or the reference's exact 50 Hz setting.
- Preserve 20 Hz policy / 120 Hz physics, 10-frame student observation history, task rewards including hard success bonus, assets, randomization, XY goal sampling and analytic soft continuation. A discontinuous success event still has no direct local orientation derivative; dense reward and learned value remain its differentiable proxies.
- Fresh two-GPU training, 2048 environments per rank, shared gradients before clipping/Adam. WM logical batch 2048/rank; actor batch 4096/rank. Actor microbatch 2048 accommodates H=16; it does not reduce logical batch size or environment count. Portable export and resume accept only schema 12; legacy checkpoints require their retained original source.

Validation includes bounded motor gradients versus old hard clipping, pre-tanh exploration, stored action semantics, no real-TD critic sampling, fixed lambda targets, independent survival-product returns and derivatives, timeout/count boundaries, WM-only warmup, BF16/FP32 restore, and two-process gradient/normalizer/model equality. Fifty-four tests passed on the first run; one legacy-error assertion differed only in message capitalization and passed after correction (55 total). Full-size Newton validation and launch results are recorded below when complete.

Fresh-stream diagnostics retain historical per-component RMSE and now explicitly add position-vector RMSE and SO(3) geodesic RMS at 1/8/16/32 steps. Matching response derivatives and real drop/success statistics still require measurement after learning; none of the initial checks establishes improved policy performance. No confidence gate, new learned reward, teacher or distillation stage is introduced.

Source references: [SuperMimic paper](https://arxiv.org/abs/2610.28480), [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed), [go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic). Both GitHub projects are reimplementations.

### Bounded20 validation and deployment

Full Newton test: two RTX 5090 GPUs, 2048 environments each, full network widths and logical batches, H=16, k=8, three joint iterations. Each rank completed 30 WM / 3 actor / 60 critic optimizer updates. Final three-network/buffer digest matched across ranks: `e01ffee83dba36c59b402a100c8bba4826da1922d3030cf00550c11a31a17115`. Checkpoint parameters and Adam states were all finite. The FP32 actor export passed checks with batch sizes 1 and 7.

Network update times were 23.426 seconds for the first compilation/update, then 1.646 and 1.643 seconds; these exclude real simulator collection. Peak sampled total GPU memory was 18,732 MiB on GPU1 (including the preserved 650 MiB viewer), and 18,112 MiB on GPU3. No runtime warnings or contact-capacity errors appeared. Maximum reward discrepancy across ranks was 9.54e-7, EMA discrepancy 2.38e-7 and hand FK position error 7.62e-6 m. This is integration validation, not a policy-quality benchmark.

The preceding schema-11 run stopped gracefully at iteration 5744 with its final checkpoint retained. New production root on server 22031: `/home/zhangke/viserDex_supermimic_bounded20_20261025`; output `outputs/revo3_bounded20_2x2048_scratch_seed42`; status `outputs/active_bounded20_scratch.json`. Production source commit is `485664e` (local implementation commit `91c6c6e`). The production command has no resume, transfer, smoke or joint-test override. Both ranks started with digest `38da9af13ffb9f1acefef3b324757d7f0b9dcc376e078b16c573261f697551a2`, distinct from the diagnostic's final model. Asset fingerprints, observation protocol, context schema, geometry digest, environment contract and noise configuration match the preceding run.

Evidence in the new remote root: `evidence/tests.log`, `evidence/tests_recheck.log`, `evidence/bounded_validation_verification.json`, `evidence/validation_state.json`, `evidence/bounded_scratch_launch.json`, and the per-rank production configuration/metrics files. The run guard monitors the two selected GPUs and does not automatically resume or restart a failed job.

## Historical Geometry20 investigation

This is an isolated optimization branch (`codex/revo3-geometry20`). It is an engineering change to the previous Geometry-v1 SuperMimic implementation, not a claim of matching the authors' original hyperparameters.

## Evidence before modifying training

The frozen parent checkpoint is Geometry-v1 iteration 21000. A 128-environment, seed-43, 640-step diagnostic with training exploration (0.1) collected 307 completed episodes, 289 drops and 714 goals. This short diagnostic is **not** directly comparable to the 4096-environment rolling training mean. It measures 26.56% action-component saturation. It does not establish saturation as the cause of drops.

On fresh real-action sequences, the frozen model's object-position component RMSE was 1.68 mm at one step, 10.64 mm at eight, 27.18 mm at sixteen and 77.96 mm at thirty-two. Predicted rewards were optimistic at eight and sixteen steps, but pessimistic at thirty-two. The old critic had no real-transition return target. These observations motivated the changes below; they do not prove that any single change will improve the final policy. An earlier diagnosis also treated fitting imagined states after a predicted drop as a bug. That claim was incorrect: both reference implementations fit the complete imagined sequence and use soft continuation in the return targets.

Diagnostic evidence on server 22021: `evidence/diagnosis30hz.json`. The diagnostic was executed using the parent 30 Hz code before deployment of this branch. The old training was subsequently stopped gracefully at iteration 21240; its full checkpoint and logs remain in the old directory.

## Time and task semantics

- Official Isaac Lab launcher and Newton MJWarp; Isaac Sim 6.2 / Newton 1.6.1 / Warp 1.17. Same physical assets, contact configuration, masses, joint limits and domain randomization.
- Physics remains 120 Hz. Policy decimation is 6 (20 Hz). Episodes remain 20 seconds (400 controls).
- Success event reward stays 250/30 = 8.3333 per goal: the manager's coefficient is now 166.6667, since it multiplies by control dt.
- EMA alpha is `1 - (1 - .95)**1.5 = .9888196601`, preserving the continuous-time filter pole.
- Action-rate coefficient is `.01 * (20/30)**2`, preserving the integrated derivative cost for the same continuous action trajectory.
- Gamma is `.99**1.5`, lambda is `.95**1.5`; other continuous reward coefficients remain unchanged. No additional drop reward/penalty is introduced.
- Actor input remains 10 x 124 noisy observations; this now spans 0.5 seconds instead of 0.333. Four-action history now spans 0.2 seconds. Sensor corruption settings remain specified in control steps: maximum perception update period of three steps is now 150 ms; one-step joint delay is 50 ms. These are intentional consequences of retaining the existing discrete observation protocol, not identical wall-clock noise timing.

The exact bound environment is generated by `scripts/supermimic/make_20hz_config.py`; the runtime binding verifies it and asset fingerprints.

## Learning changes

1. Keep the six physical groups and five geometric supervision groups, actual-asset geometry, and eight-step autoregressive WM fit.
2. Actor imagination is eight steps (0.4 seconds), down from thirty-two at 30 Hz (1.067 seconds). A learned value bootstrap supplies the remaining return.
3. Actor and imagined critic share `c_t = sigmoid((0.3 - norm(predicted_object_position)) / 0.01) * (count_t < 50)`. Keep the full imagined sequence across predicted drops, including gradients beyond 0.3 m. Only known time/count limits mask later imagined states. Timeouts bootstrap once with the same continuation; true simulator physical termination stays hard in real TD targets.
4. Train the critic with independently normalized 50% imagined lambda-return loss and 50% real TD(0) loss. Hold targets fixed during the twenty critic updates. An EMA target critic (tau .05 per update) supplies target values. Actor gradients still pass through the online critic and frozen WM.
5. Actor learning rate is one quarter of the WM/critic rate (nominal 5e-5 versus 2e-4), with the existing 1000-update ramp. The existing gradient clipping and exploration settings remain.
6. Transfer only the parent actor and its observation normalizer. Start the WM, online/target critics, Adam optimizers, replay and counters afresh, because 30 Hz deltas are not 20 Hz deltas. Freeze both actor weights and its normalizer for the first 500 iterations while fitting the WM and real TD critic.
7. Checkpoint schema 11 records `soft_drop_continuation_actor_and_critic_v1`. A schema-10 checkpoint requires explicit `--resume ... --upgrade_continuation`, restoring all four networks, three optimizers and training counters. This changes the learning objective without reinitializing networks. Simulator/replay are warmed afresh as with existing resume. The original 30-to-20 Hz actor transfer remains a separate operation; it is not repeated for this correction.

Every 200 iterations an eager frozen WM snapshot predicts the next 32 real transitions. Its snapshot is taken before fitting these new observations. Report steps 1/8/16/32, mask after real episode boundaries and log action saturation. Data may enter training after observation: this is a fresh-stream diagnostic, not a permanently held-out dataset.

## Validation and deployment

Tests cover reward/filter/discount time conversion, terminal masks, real TD timeout behavior, target EMA, frozen actor calibration, pre-reset replay state, fixed targets, geometry, BF16 checkpoint restore and two-rank gradients/normalizers/model equality. Full-size Newton validation uses two GPUs with 2048 environments each, forces timeouts, audits real reward/EMA/observations/FK, exercises the actor and both critic sources, then checks final network digests and saves a diagnostic checkpoint. The production run has a distinct output directory and does not resume the diagnostic model.

Validation server: 22021. Production server: 22031, physical GPUs 1 and 3. The existing Revo3 joint viewer (PID 1900441, about 650 MiB, idle) remains on GPU 1; no training job shares these GPUs. The attempted production launch on 22021 was refused after another user occupied GPU 0. New root: `/home/zhangke/viserDex_supermimic_geometry20_20261025`. Active state: `outputs/active_geometry20.json`. Production output: `outputs/revo3_geometry20_2x2048_actor21000_seed42`. Old output remains under `/home/zhangke/viserDex_supermimic_geometry_20261024`.

Judge improvement by matched-seed evaluation over equal simulated seconds, goals per completed episode, drops divided by completed episodes, and goals per simulated second. Neither initial calibration losses nor different-frequency iteration counts establish improvement.

Validation results: 39 tests passed (31 existing contract/geometry, 7 new regression, 1 two-rank test). Full-size Newton validation completed 3 joint iterations with 30 WM, 3 actor and 60 critic optimizer updates per rank; both ranks produced digest `49e3156ec941d23f7fcfc9bdb1953fdcadc495b1c55f0004ecf2dad8621b6470`. Reward audit max error 9.54e-7. Steady network update times in this short validation were 1.62 and 1.58 seconds (excluding simulator collection; initial compilation took longer). Portable FP32 actor export passed batch-size checks.

Old training reference (iterations 20741–21240): 14711 completed episodes, weighted drop fraction 0.705594, 5.04249 goals per completed episode. These use training exploration, and are not a matched evaluation of the 20 Hz policy.

Production transfer on 22031 uses an actor-only extraction of the parent checkpoint with the original checkpoint SHA recorded in `source_checkpoint_sha256`. Its actor weights, observation normalizer, protocol metadata and noise configuration are unchanged. The robot and object root USD hashes and the official runtime package versions were checked against the validated 22021 setup before launch.


## Soft continuation correction (2026-10-25)

The initial schema-10 implementation combined sigmoid survival with a hard distance cutoff for the actor, but used only the hard cutoff for imagined critic targets. This was a mismatch introduced in the 20 Hz change. The preceding 30 Hz implementation already used soft continuation. Schema 11 fixes this mismatch; it does not introduce a learned termination classifier.

With `C_0 = 1`, `C_(t+1) = C_t * c_t`, the actor maximizes `sum_t gamma^t C_t r_t + gamma^H C_H V(s_H)`. The current transition reward is weighted by `C_t`; `c_t` discounts future rewards and the value bootstrap. The critic target is `G_t = r_t + gamma c_t ((1-lambda) V_target(s_(t+1)) + lambda G_(t+1))`. The two use the same continuation, although online and EMA target value networks intentionally differ. At a timeout, fold `gamma c_t V(s_(t+1))` into the final reward and stop the sequence. A simultaneous true count termination suppresses this bootstrap. Real replay retains the simulator's hard `terminated` bit and captured pre-reset state.

Reference code, both unofficial implementations:
- [zita-ch/supermimic_vibed train.py](https://github.com/zita-ch/supermimic_vibed/blob/main/supermimic/train.py): cumulative soft survival in the actor and soft continuation in lambda returns; all imagined states train the critic.
- [zita-ch/supermimic_vibed mdp.py](https://github.com/zita-ch/supermimic_vibed/blob/main/supermimic/mdp.py): analytic termination from predicted physical state.
- [Zhengtianjin/go2-supermimic imagine.py](https://github.com/Zhengtianjin/go2-supermimic/blob/main/go2_supermimic/supermimic/imagine.py): analytic fall probability, cumulative survival and continuation-weighted returns. Its separate termination reward is task-specific and is not copied into Revo3.

This correction retains the current 20 Hz / 120 Hz control/physics periods, eight-step imagination, eight-step WM fitting, assets, rewards, noise, 50/50 real/imagined critic loss, target-critic EMA and learning rates. Soft continuation provides a differentiable learning signal; it does not guarantee a lower real drop rate. That requires matched evaluation.

Deployment of this correction is isolated at `/home/zhangke/viserDex_supermimic_geometry20_soft_20261025`, on server 22031. The preceding schema-10 source, run and final checkpoint remain intact. The production state is `outputs/active_geometry20_soft.json` in the new root.


Correction validation: 51 tests passed, including independent survival-product actor returns, lambda returns as a mixture of n-step returns, gradients on both sides of the drop boundary, timeout/count precedence, explicit v10 migration preserving network/Adam state, and two-rank synchronization. Full-size Newton validation resumed iteration 768, performed three joint updates (30 WM / 3 actor / 60 critic), and finished at 771. Initial four-network digest matched the parent on both ranks; final digest was `7526e20b6257d8bd0f2b14708ff59152e86ad3a052939247df708ff0658a22af`. All checkpoint network and optimizer tensors were finite. Reward audit error was at most 9.54e-7; FK position error at most 7.52e-6 m. Steady network update times were 1.343 and 1.328 seconds, excluding simulator collection. Schema-11 FP32 actor export passed batch-size 1 and 7 checks.

Production resumes the original iteration-768 checkpoint, retaining its 7680 WM / 268 actor / 15360 critic updates. The diagnostic iteration-771 checkpoint is not used for production. Source code commit on server 22031 is `60567d5`; local equivalent is `c187046`. Evidence: `evidence/regression_test_result.json`, `evidence/soft_continuation_verification.json`, and `evidence/soft_production_launch.json` in the corrected remote root. Current output is `outputs/revo3_geometry20_soft_2x2048_resume_seed42`.


## User-requested fresh start (2026-10-25)

The user subsequently requested training from zero. The resumed run was saved and stopped at iteration 994. `scripts/supermimic/launch_soft_scratch.py` starts one jointly updated model on server 22031 GPUs 1 and 3, with 2048 environments per rank, 4096 total and 20 Hz control. It supplies neither a resume checkpoint nor actor-transfer weights: networks, normalizers, Adam state, replay and counters start fresh. All training hyperparameters match the prior corrected run, including the initial 500 iterations of WM/real-critic calibration before joint policy updates.

Current run: `outputs/revo3_geometry20_soft_2x2048_scratch_seed42`. Current state: `outputs/active_geometry20_soft_scratch.json`. The previous resume directory and final checkpoint are retained. Launch and startup checks are recorded in `evidence/soft_scratch_launch.json` and `evidence/soft_scratch_verification.json`.
