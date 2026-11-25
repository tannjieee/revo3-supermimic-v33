"""Run the same distributed contract checks on the actual two training GPUs."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/viserdex"), str(ROOT / "tests")]
import torch
from viserdex.supermimic.distributed import Distributed
from test_supermimic_distributed import distributed_checks

device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
distributed = Distributed.initialize(device)
output = Path(sys.argv[1])
if distributed.rank == 0:
    output.mkdir(parents=True, exist_ok=False)
distributed.barrier()
try:
    distributed_checks(device, output)
    print(f"NCCL SuperMimic rank {distributed.rank} PASS", flush=True)
finally:
    distributed.close()
