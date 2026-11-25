# Revo3 EventRamp v1 — implementation and validation status

Independent task: `Isaac-ViserDex-Repose-Revo3-SuperMimic-EventRamp-v1`.
Independent branch: `codex/revo3-event-ramp`. This copy does not register the baseline ID.

## State decision

Retain physical output 55 (54 tangent delta), full state 172, actor 124×10 and context 548. The 55 coordinates alone omit drive memory and domain parameters; the WM already receives the EMA target and context. The implicit position controller in assets/revo3.py has no configured learned actuator hidden state or action delay. inhand_env_cfg.py randomizes mass, materials and gains at startup, and resets pose/joints at reset; no interval force event was found. Fixed geometry plus joints and object pose determines geometric separation, and velocities determine instantaneous relative surface motion. These do not uniquely determine contact impulse or solver warm-start state. Actual engine internal state sufficiency remains unverified until remote testing. Base-relative velocity assumes the configured fixed hand base; moving-base generalization would require base twist/orientation and correct relative velocities. No claim of strict engine Markov sufficiency is made.

Do not predict static context, redundant FK, reward, termination, or clean actor observations. Goals/time/count/action history remain analytic; sensor memory remains simulated. No teacher policy is loaded. No changes to reward, physics, sensor protocol or actor interface. No contact auxiliary labels until a valid engine contact recorder is available. A recurrent or history WM would need a reset-aware history contract in replay AND imagination and additional evidence of residual hidden dynamics; it is not enabled speculatively.

## Changes

CPU archive holds at most 4096 compact 8-step windows (state, next physical, action, done and env ID), without observation histories or sensors. At most 16 candidate environments are admitted per real step, randomly selected. Events are termination, object distance >0.20 m, speed >1 m/s, goal angle <0.1 rad, or recovery proxy (distance >0.15 m followed by >0.02 m decrease). These are heuristic motion/outcome events, not contact or true slip labels. Windows with any earlier done are rejected; a done in the last transition remains valid. Overlapping windows can occur and event categories are not balanced. Archive sampling replaces the first 25% of WM samples when available, with replacement; empty archive uses ordinary replay. Actor starts use ordinary replay. Each rank keeps its own archive and uses the baseline gradient/statistic synchronization. Archive adds about 31 MiB/rank at capacity; replay histories remain the dominant allocation. Resume rewarms both replay and archive, matching baseline semantics.

Keep 500 WM-only iterations. First actor update uses 1/1000 of the scheduled LR; update 1000 reaches the scheduled LR. WM/Critic cosine schedules are unchanged. Config supports event_fraction=0 and actor_ramp_iters=0 for ablation.

Existing `--holdout` collects 32 new real steps after fitting, excludes post-reset transitions, and now reports all prediction horizons 1–32 with six physical RMSE groups and action-conditioned real/predicted reward means and MAE. This is trajectory-conditioned reward validation, not a counterfactual on-policy return estimate or evidence of convergence. Holdout trajectories are never added to replay or normalizer updates.

## Remote validation (22021)

Connection recovered by explicitly disabling agent/key authentication (`look_for_keys=False, allow_agent=False`) and using the provided passwords. The previous connection block is resolved. GPU0 UUID GPU-f5aba079-86bd-2df0-cac5-68663b176dbb and GPU4 UUID GPU-ac561d84-ce6e-f1ac-82ee-8c9dc3316811 were rechecked idle before validation. Baseline GPU1/5 PIDs 175860/175861 remained running and all 16 remote source fingerprints matched.

Official existing runtime passed all 25 CPU contract tests in 9.90s. Real full-batch two-rank validation used 2048 environments per rank, WM batch2048, actor4096, critic microbatch32768, BF16, forced timeout audit and 32 fresh holdout steps. Three diagnostic joint iterations completed: WM30/Actor3/Critic60 optimizer steps per rank; final model digests matched. Real reward max error 2.235e-8, EMA error 2.384e-7. Both ranks wrote validation.json and holdout.json. Actual full-size checkpoint actor export/reload parity passed.

This diagnostic intentionally skips WM-only warmup and is separate from production. Initial held-out predictions are poor at long horizons: rank0 object position RMSE 0.00168m at step1 and 0.0621m at step32, where persistence gives 0.00228m. Orientation error grows from 0.0205rad to 0.531rad. This is evidence that training is needed, not proof that archive/ramp improves learning. Reward MAE grows from 0.000126 to 0.00236. No rollback/stop threshold is enabled.

Deployment root: /home/zhangke/viserDex_supermimic_event_ramp_20261024. Production run: outputs/revo3_event_ramp_2x2048_scratch_seed42. Independent monitor: outputs/active_event_ramp.json. Production retains seed42/43, 500 WM-only iterations, 320000 iteration budget, all baseline environment/reward/observation settings. Checkpoints every2000 and latest every100 iterations. Source snapshots and full evidence are stored alongside this repository.

## Local verification

25 unique CPU tests passed, including original checkpoint/gradient/normalizer tests, event/reset tests, holdout reward masks and two-process Gloo model/statistic synchronization. Synthetic version-3 checkpoint actor export and reload parity passed for batches 1 and 7. Local Python 3.12 / Torch 2.14.1 differs from the server runtime; this does not validate CUDA, NCCL, physics or full production batches. Baseline 16-file SHA-256 recheck passed.

## Production confirmation

Both ranks advanced to iteration153 in WM-only phase. The iteration100 latest.pt checkpoint (139726525 bytes) loaded successfully: WM1000/Actor0/Critic0 optimizer steps; parent_checkpoint=None; new task ID verified. Both ranks progressed beyond the collective model equality check at100. Rank0 records digest 2e6dcefcb6303aa9452c4e24aa04b0b5b04a2ef71c6a694292fdbadf67f73254; baseline trainer writes this metric only in rank0 logs. Remote source revision f5aad425f496bd006b9a8b3dfbb8dd472a328c1f. New training PIDs200063/200064, independent guard200036. Remote baseline source fingerprints still match. This confirms startup/checkpoint health, not convergence or improvement.
