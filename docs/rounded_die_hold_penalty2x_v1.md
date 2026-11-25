# RoundedDie Hold-Penalty2x-v1

Independent copy of Hold-v1, task ID
`Isaac-ViserDex-Repose-Revo3-SuperMimic-RoundedDie-Hold-Penalty2x-v1`.

All four motion penalties are doubled, before control-dt scaling:

| Term | Hold-v1 | This run |
|---|---:|---:|
| Joint velocity squared | -0.00003125 | -0.0000625 |
| Action magnitude squared | -0.000125 | -0.00025 |
| Action first difference squared | -0.005555555556 | -0.011111111111 |
| Action second difference squared | -0.005555555556 | -0.011111111111 |

The real reward manager and differentiable imagined reward use matching weights.
Nominal finger Kp is reduced from 7 to 3; Kd stays 0.22. Existing stiffness
randomization scales Kp by log-uniform [0.6, 1.5], giving actual Kp [1.8, 4.5].
EMA stays 0.5, control stays 20 Hz, hold stays 0.1 rad / 0.5 rad/s / 0.5 s.
No hold-escalation or new reward term is added. Assets, observations,
randomization, network widths and optimizer schedule are inherited unchanged.
World model, Actor and Critic start from random initialization; no old policy,
optimizer or replay is loaded. Production uses two GPUs, 2048 environments each,
500 world-model-only warmup iterations and the existing Actor learning-rate ramp.
Diagnostic checkpoints are never used to initialize production.

Generate the new binding with `scripts/supermimic/prepare_hold_penalty2x.py`.
Launch with `scripts/supermimic/start_hold_penalty2x.py --start-training
--gpu GPU_UUID_A --gpu GPU_UUID_B` after both GPUs are idle.
The output is `outputs/rounded_die_hold_penalty2x_2x2048_scratch_seed42`;
state is `outputs/active_rounded_die_hold_penalty2x.json`.
Old Hold-v1 remains in its original repository with its final saved checkpoint.
