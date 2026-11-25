# ViserDex migration policy

The user requires the official Isaac Lab v3.0.0 Newton migration workflow.
Use Isaac Sim 6.2.0.0, Newton 1.6.1 and Warp 1.17.0 with the official release's
workspace, importer extra and dependency overrides. Follow the pinned official
migration, asset preparation and MJWarp tuning guides linked in
`docs/newton_migration.md` before changing engine-specific behavior.

Keep the default `physics=newton_mjwarp` preset. Use official `launch_simulation`
and unified `isaaclab train/play` entrypoints. Put solver differences in presets
and schema fragments; preserve task observations, joint order and reward logic.
Generate robot assets through `scripts/preprocessing/prepare_revo3_newton.py`.
Never edit vendored BrainCo inputs or disable randomization to hide a migration
failure. Check live collision geometry and reject contact-capacity warnings.
Record actual validation and any remaining limits in the migration document.
