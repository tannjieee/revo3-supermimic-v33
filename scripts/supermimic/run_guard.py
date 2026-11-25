"""Keep a detached training job's PID, checkpoint, metrics and GPU state inspectable."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--gpu", required=True, action="append", help="Repeat once per physical GPU")
p.add_argument("--allowed_idle_viewer_pid", type=int, action="append", default=[])
p.add_argument("--run_dir", type=Path, required=True)
p.add_argument("--state_file", type=Path, required=True)
p.add_argument("--console", type=Path, required=True)
p.add_argument("command", nargs=argparse.REMAINDER)
args = p.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
query = [
    "nvidia-smi",
    "--id=" + ",".join(args.gpu),
    "--query-gpu=uuid,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
]


def gpu_rows(output):
    rows = {}
    for line in output.strip().splitlines():
        uuid, used, utilization = (x.strip() for x in line.split(","))
        rows[uuid] = dict(memory_mib=int(used), utilization_percent=int(utilization))
    return rows


initial_gpus = gpu_rows(subprocess.check_output(query, text=True))
allowed = set(args.allowed_idle_viewer_pid)
for pid in allowed:
    command_line = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode()
    if "revo3_joint_viewer_20261021" not in command_line or "--viz viser" not in command_line:
        raise RuntimeError("Only the identified existing Revo3 joint viewer may share a GPU")
compute_rows = subprocess.check_output(
    ["nvidia-smi", "--id=" + ",".join(args.gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
compute_pids = {int(x.strip()) for x in compute_rows.splitlines() if x.strip()}
limit = 1024 if allowed else 500
if (len(initial_gpus) != len(args.gpu) or compute_pids - allowed
        or any(row["memory_mib"] >= limit or row["utilization_percent"] > 5 for row in initial_gpus.values())):
    raise RuntimeError(f"Requested GPU is occupied or repeated; refusing launch: {initial_gpus}")
used = max(row["memory_mib"] for row in initial_gpus.values())
if args.run_dir.exists():
    raise FileExistsError(args.run_dir)
args.state_file.parent.mkdir(parents=True, exist_ok=True)
args.console.parent.mkdir(parents=True, exist_ok=True)
state = dict(
    status="starting",
    gpu_uuid=args.gpu[0],
    gpu_uuids=args.gpu,
    gpus=initial_gpus,
    peak_memory_per_gpu_mib={key: row["memory_mib"] for key, row in initial_gpus.items()},
    run_dir=str(args.run_dir),
    command=command,
    console=str(args.console),
    started_unix=time.time(),
    peak_gpu_memory_mib=used,
)


def save_state():
    state["updated_unix"] = time.time()
    tmp = args.state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(args.state_file)


with args.console.open("x") as log:
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state["launcher_pid"] = proc.pid
    state["guard_pid"] = os.getpid()

    def stop(signum, frame):
        state["stop_requested"] = True
        if proc.poll() is None:
            # The exact Python PID comes from this run's metadata, avoiding a
            # broad process-name kill and allowing train.py's final save.
            pid = state.get("training_pid")
            if pid is not None:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                os.killpg(proc.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save_state()
    while proc.poll() is None:
        config_file = args.run_dir / "config.json"
        if config_file.exists() and "training_pid" not in state:
            try:
                metadata = json.loads(config_file.read_text())["metadata"]
                state["training_pid"] = metadata["pid"]
                state["world_size"] = metadata.get("world_size", 1)
            except (ValueError, KeyError):
                pass
        pids = {}
        for rank in range(state.get("world_size", 1)):
            path = args.run_dir / (f"rank_{rank}/config.json" if rank else "config.json")
            if path.exists():
                try:
                    pids[str(rank)] = json.loads(path.read_text())["metadata"]["pid"]
                except (ValueError, KeyError):
                    pass
        state["training_pids"] = pids
        metrics_file = args.run_dir / "metrics.jsonl"
        if metrics_file.exists():
            with metrics_file.open("rb") as stream:
                stream.seek(max(0, metrics_file.stat().st_size - 32768))
                lines = stream.read().splitlines()
            for line in reversed(lines):
                try:
                    state["metrics"] = json.loads(line)
                    state["status"] = "running"
                    break
                except ValueError:
                    continue
        ckpt = args.run_dir / "latest.pt"
        if ckpt.exists():
            state["latest_checkpoint"] = str(ckpt)
            state["checkpoint_mtime"] = ckpt.stat().st_mtime
        result = subprocess.run(query, capture_output=True, text=True)
        if result.returncode == 0:
            rows = gpu_rows(result.stdout)
            state["gpus"] = rows
            for key, row in rows.items():
                state["peak_memory_per_gpu_mib"][key] = max(
                    row["memory_mib"], state["peak_memory_per_gpu_mib"].get(key, 0)
                )
            used = max(row["memory_mib"] for row in rows.values())
            state["gpu_memory_mib"] = used
            state["gpu_utilization_percent"] = max(row["utilization_percent"] for row in rows.values())
            state["peak_gpu_memory_mib"] = max(used, state["peak_gpu_memory_mib"])
        save_state()
        time.sleep(3)
    state["exit_code"] = proc.wait()
state["status"] = "stopped" if state.get("stop_requested") else ("completed" if proc.returncode == 0 else "failed")
state["finished_unix"] = time.time()
save_state()
raise SystemExit(proc.returncode)
