# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'viserdex' python package."""

from pathlib import Path
import tomllib

from setuptools import find_namespace_packages, setup

ROOT = Path(__file__).resolve().parent
with (ROOT / "config" / "extension.toml").open("rb") as stream:
    PACKAGE = tomllib.load(stream)["package"]

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "psutil>=5.9.8",
    "trimesh",
    "pandas",
    "h5py",
    "Pillow",
    "PyYAML",
    "scikit-image",
    "open3d",
    "plyfile",
    "kornia",
    "numpy>=2,<3",
    "ninja",
    "termcolor",
    "tensorboard",
    "wandb",
    "gsplat==1.5.3",
    "viser==0.2.11",
]

# Installation operation
setup(
    name="viserdex",
    packages=find_namespace_packages(
        where="source/viserdex",
        include=["viserdex*"],
        exclude=[
            "viserdex.assets.data",
            "viserdex.assets.data.*",
            "viserdex.assets.generated",
            "viserdex.assets.generated.*",
        ],
    ),
    package_dir={"": "source/viserdex"},
    author=PACKAGE["author"],
    maintainer=PACKAGE["maintainer"],
    url=PACKAGE["repository"],
    version=PACKAGE["version"],
    description=PACKAGE["description"],
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    keywords=PACKAGE["keywords"],
    project_urls={
        "Upstream": "https://github.com/leggedrobotics/ViserDex",
        "Issues": "https://github.com/star3000times/viserDex_revo3/issues",
    },
    install_requires=INSTALL_REQUIRES,
    entry_points={"isaaclab.tasks": ["viserdex = viserdex.tasks"]},
    license="BSD-3-Clause",
    license_files=["LICENCE", "THIRD_PARTY_NOTICES.md"],
    include_package_data=True,
    package_data={"viserdex": ["assets/brainco/**/*", "assets/*.usda", "supermimic/data/fit_03/*.json"]},
    python_requires=">=3.12,<3.13",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.12",
    ],
    zip_safe=False,
)
