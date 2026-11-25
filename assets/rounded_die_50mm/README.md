# 50 mm original-shape die, nominal 100 g

Selected by the user on 2026-11-05. This is the accepted original 70 mm mesh
uniformly scaled by 5/7, with the same face numbering and yellow / midnight-green
appearance as `assets/dice_50mm_from_original_70mm` in the outer workspace.
The rejected rebuilt print design is not used.

`object_newton.usda` is a self-contained SI, Z-up asset. Both the 11206-triangle
visual and the original 200-vertex outer convex collision mesh are scaled in
vertex coordinates; the rigid root and runtime spawn scale remain unity.
Pip recesses remain visual concavities under the existing outer-hull collision
approximation. The original body frame is preserved, rather than the print's
bottom-centered frame. Original 70 mm assets remain unchanged.

The user specifies nominal mass **0.1 kg**. Inertia is an estimate from the same
shape / relative mass distribution:

```
I50 = I70 * (0.1 / m70) * (5/7)^2
```

Principal axes are unchanged; center of mass is scaled by 5/7. This is not a
measurement of the print's internal infill or inertia. Existing mass
randomization remains 0.8–1.2 times nominal, so randomized worlds use 80–120 g.
Friction randomization, hand, actuators, timing, thresholds, and health checks
are unchanged.

Reproduce into a new output directory using the policy Python environment:

```bash
python scripts/preprocessing/prepare_die_50mm.py \
  --accepted_source /home/brainco/reorientation/assets/dice_50mm_from_original_70mm \
  --mass_kg 0.1 --output /path/to/new/output
```

The generator verifies the accepted print source mesh hashes before generating.
`manifest.json` records lineage, dimensions, nominal mass, inertia and USD hash.
Two independent generations produced identical USD bytes. Source attribution
is preserved in `ATTRIBUTION.txt`.

New manual-convex training defaults to `--object_variant die50`. A saved run
retains its own asset; explicitly requesting `die50` with a saved 70 mm
environment is rejected. Geometry and environment contracts also prevent a
70 mm model from silently resuming as a 50 mm model. Details and finite Newton
validation: `docs/versions/v30_video_die50_20261105/`.
