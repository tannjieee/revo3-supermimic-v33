# 50 mm / 47 g die

User-specified nominal mass: **0.047 kg**. Geometry is unchanged from the accepted
50 mm version. `die50_47g` is the fresh-run default; saved `die50` runs still refer
to the separate 100 g asset.

The inertia tensor is scaled by 0.47 from the 100 g version, assuming the same
relative internal mass distribution. It is not a measured printed-die inertia.
Existing mass randomization remains enabled; 47 g is the nominal value.

See `manifest.json` for source hashes and dimensions, and `ATTRIBUTION.txt` for
the original mesh attribution. Reproduce from the accepted source with:

```bash
python scripts/preprocessing/prepare_die_50mm.py \
  --accepted_source /home/brainco/reorientation/assets/dice_50mm_from_original_70mm \
  --mass_kg 0.047 --output assets/rounded_die_50mm_47g
```
