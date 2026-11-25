# Third-party notices

The repository combines code, robot descriptions, and separately downloaded data.
The root code license does not grant rights to every asset or dependency.

## ViserDex code

- Source: [leggedrobotics/ViserDex](https://github.com/leggedrobotics/ViserDex).
- License: BSD-3-Clause; the original notice is retained in [LICENCE](LICENCE).
- Upstream authors and contributors are retained in source headers and [CONTRIBUTORS.md](CONTRIBUTORS.md).
- This fork adds Revo3 integration and training/scanning tooling. It is maintained separately from the upstream project.

## BrainCo Revo3 descriptions, meshes, and textures

- Source: [BrainCoTech/brainco-description](https://github.com/BrainCoTech/brainco-description/tree/f197d9d720b1b106a3298e2438ff07e055bf15ec).
- Revision: `f197d9d720b1b106a3298e2438ff07e055bf15ec`, release `20261009`.
- Bundled scope: `source/viserdex/viserdex/assets/brainco/`.
- Provenance and SHA-256 checksums: [UPSTREAM.json](source/viserdex/viserdex/assets/brainco/UPSTREAM.json).

The bundled [upstream README](source/viserdex/viserdex/assets/brainco/README.md#license)
says license information will be provided with the public release. No explicit
asset license accompanies this snapshot. These files are not relicensed under
BSD-3-Clause. Confirm permission with the upstream rights holder before publicly
redistributing or modifying this asset bundle. Public availability alone is not a
license grant.

The project's `assets/revo3_right_*.usda` files are local integration layers;
they reference the BrainCo descriptions and do not change ownership of the underlying assets.

## Object dataset

[leggedrobotics/ViserDexSplats](https://huggingface.co/datasets/leggedrobotics/ViserDexSplats)
provides the object data. This checkpoint branch includes the RSLCube subset in
`source/viserdex/viserdex/assets/data/` for reproduction; the remaining objects
are downloaded separately. Source revision and checksums are recorded in
`assets/data/assets-source.json`. Generated cube assets derive from that dataset.
The dataset is identified by the upstream project as CC BY-NC 4.0: attribution and
non-commercial terms apply separately from the code license. Retain dataset notices
and check the downloaded revision's terms when using or redistributing it.

Locally scanned objects and trained weights need their own provenance and
redistribution terms; they are not automatically covered by the root license.

## Dependencies

Collected dependency license texts are in [docs/licenses/](docs/licenses/).
These notices are attribution records, not a replacement for each installed
version's terms. Isaac Sim, Isaac Lab, PyTorch, CUDA, gsplat, COLMAP, Open3D, and
other dependencies remain subject to their own licenses. Runtime dependencies
are installed separately rather than vendored into the Python wheel.
