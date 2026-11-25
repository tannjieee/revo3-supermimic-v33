#!/usr/bin/env python3
"""Standalone scanning entry point; does not import Isaac Sim."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/viserdex"))

from viserdex.scanning.cli import main

if __name__ == "__main__":
    main()
