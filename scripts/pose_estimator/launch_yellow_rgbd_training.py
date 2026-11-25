"""Launch only the isolated offline predictor on the two explicitly reserved GPUs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name:
        raise ValueError("run_name must be one directory name")
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    os.nice(10)
    evidence = root.parent / "evidence" / args.run_name
    evidence.mkdir(parents=True, exist_ok=False)
    state_path = evidence / "guard.json"
    state = dict(status="checking", guard_pid=os.getpid(), started_unix=time.time(), smoke=args.smoke)

    def write():
        temp = state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(state_path)

    lock = (root.parent / "evidence/yellow_fusion_gpu24.lock").open("a")
    write()
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpus = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"
        ], text=True)
        expected = {2: "GPU-c353bd61-5a0a-74b4-de6b-91243ff9931b",
                    4: "GPU-ac561d84-ce6e-f1ac-82ee-8c9dc3316811"}
        found = {}
        for row in gpus.splitlines():
            index, uuid, memory = [x.strip() for x in row.split(",")]
            if int(index) in expected:
                if expected[int(index)] != uuid or int(memory) > 256:
                    raise RuntimeError(f"Requested GPU not idle or identity changed: {row}")
                found[int(index)] = uuid
        if found != expected:
            raise RuntimeError("Requested GPU missing")
        processes = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"
        ], text=True)
        if any(uuid in processes for uuid in expected.values()):
            raise RuntimeError("Requested GPU already has a compute process")
        dataset = root / "outputs/yellow_rgbd_20k_train_1k_val_2x1000_20261102_v2"
        manifest = json.loads((dataset / "manifest.json").read_text())
        original = Path("/home/zhangke/viserDex_supermimic_yellow_die_20261101/repository")
        hashes = {p: hashlib.sha256((original / p).read_bytes()).hexdigest()
                  for p in manifest["protected_source_sha256"]}
        if hashes != manifest["protected_source_sha256"]:
            raise RuntimeError("Original protected files changed since capture; investigate before launch")
        output = root / "outputs" / args.run_name
        if output.exists():
            raise FileExistsError(output)
        command = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
                   "--nnodes=1", "--nproc_per_node=2", "scripts/pose_estimator/train_yellow_rgbd_fusion.py",
                   "--dataset_root", str(dataset), "--rgb_checkpoint", "checkpoints/pose_estimator/cnn_6202_final.pth",
                   "--output_dir", str(output), "--epochs", "1" if args.smoke else "2000",
                   "--batch_size", "256", "--workers", "8", "--batch_norm", "sync", "--amp", "bf16",
                   "--freeze_epochs", "0", "--seed", "42"]
        if args.smoke:
            command += ["--max_train_batches", "2"]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=",".join(expected.values()), PYTHONPATH=str(root / "source/viserdex"),
                   PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
        files = [Path(__file__), root / "scripts/pose_estimator/train_yellow_rgbd_fusion.py",
                 root / "source/viserdex/viserdex/collectors/yellow_fusion_data.py",
                 root / "source/viserdex/viserdex/collectors/rgbd_fusion_network.py",
                 root / "source/viserdex/viserdex/utils/pose_metrics.py"]
        state.update(command=command, physical_gpus=expected, output_dir=str(output),
                     protected_source_sha256=hashes, gpu_processes_before=processes,
                     code_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        with (evidence / "train.log").open("x") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            state.update(status="running", training_pid=process.pid)
            write()
            result = process.wait()
        state.update(status="completed" if result == 0 else "failed", exit_code=result, finished_unix=time.time())
        state["protected_source_hashes_unchanged"] = all(
            hashlib.sha256((original / p).read_bytes()).hexdigest() == value for p, value in hashes.items())
        write()
        if result:
            raise SystemExit(result)
    except Exception as error:
        state.update(status="failed", error=repr(error), finished_unix=time.time())
        write()
        raise


if __name__ == "__main__":
    main()
