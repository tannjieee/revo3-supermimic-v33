"""Run a command and sample total GPU memory (includes Newton/Warp, unlike Torch stats)."""

import argparse
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--gpu", required=True)
p.add_argument("--report", type=Path, required=True)
p.add_argument("command", nargs=argparse.REMAINDER)
a = p.parse_args()
command = a.command[1:] if a.command[:1] == ["--"] else a.command
proc = subprocess.Popen(command)
samples = []
start = time.monotonic()
while proc.poll() is None:
    result = subprocess.run(
        ["nvidia-smi", "--id=" + a.gpu, "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        used, util = map(int, result.stdout.strip().split(","))
        samples.append(dict(seconds=time.monotonic() - start, memory_mib=used, utilization=util))
    time.sleep(1)
a.report.parent.mkdir(parents=True, exist_ok=True)
a.report.write_text(
    json.dumps(
        dict(
            exit_code=proc.returncode,
            samples=samples,
            peak_gpu_memory_mib=max((s["memory_mib"] for s in samples), default=None),
        ),
        indent=2,
    )
)
raise SystemExit(proc.returncode)
