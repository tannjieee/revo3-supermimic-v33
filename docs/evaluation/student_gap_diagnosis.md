# Final student gap diagnosis, 2026-10-23

Read-only tests on the existing student1000checkpoint:64worlds,600steps,seed53,30Hz.
No student/teacher weights or training implementation were changed.

| Policy/input | Goals | Object out of reach |
|---|---:|---:|
| Student, full pose/joint noise |767|58|
| Student, clean pose/original joint noise |922|48|
| Student, clean pose and joints |1012|28|
| Teacher, original joint noise/privileged truth |1123|7|
| Teacher, clean joints/privileged truth |1217|7|

This supports two contributors: noise robustness and a residual control imitation/state
inference gap. It does not establish one specific implementation fault. The CNN pose
estimator is not part of these tests. Training completed its budget; it did not reach
teacher performance.

An additional shadow rollout measured38400student states without changing returned
actions. Input position vector RMSE79.16mm was reduced by the auxiliary decoder to
11.37mm; mean orientation error24.60deg became11.68deg. Normalized velocity-readout
R2:joint=.921,object linear=.596,object angular=.546. Action RMSE relative to a frozen
teacher in raw policy-output units=.849; large-observation-error states had1.044 vs
.740elsewhere. These are diagnostic associations, not isolated causal interventions.
The actor uses fused latent features, not decoder outputs directly.

Limitations:single seed, closed-loop trajectories differ across noise conditions.
The instrumented shadow rollout got786goals/46out-of-reach vs uninstrumented767/58;
its readout metrics should not be presented as an identical-trajectory replay.
Teacher forward defaults to deterministic output; source files match pre-test hashes.
All four diagnostic GPU jobs exit0, no capacity/numerical fatal event.
