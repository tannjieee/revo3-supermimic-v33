"""Wait for pose training, evaluate its best weights, then distill and evaluate Revo3.

Each stage must succeed before the next starts. Run from the repository root;
the orchestrator itself uses only the Python standard library.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose_run", type=Path, required=True)
    parser.add_argument("--pose_launch", type=Path, required=True)
    parser.add_argument("--validation_dataset", type=Path, required=True)
    parser.add_argument("--teacher_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--wait_timeout_hours", type=float, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for name in ("pose_run", "pose_launch", "validation_dataset", "teacher_checkpoint", "output_dir"):
        value = getattr(args, name)
        setattr(args, name, (root / value).resolve())
    if min(args.max_iterations, args.num_envs, args.wait_timeout_hours) <= 0:
        parser.error("Counts and timeout must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stage = "waiting_for_pose"
    child = None
    stopping = False

    def status(**extra):
        record = {"stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(), **extra}
        temporary = args.output_dir / "pipeline.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(args.output_dir / "pipeline.json")
        print("PIPELINE", json.dumps(record), flush=True)

    def stop(signum, _frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)

    def run(name, command):
        nonlocal stage, child
        if stopping:
            raise InterruptedError("Pipeline interrupted")
        stage = name
        with (args.output_dir / f"{name}.log").open("x") as stream:
            child = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            status(command=command, child_pid=child.pid)
            result = child.wait()
        child = None
        if result or stopping:
            raise RuntimeError(f"Stage {name} exited with code {result}; interrupted={stopping}")

    def read(name):
        return json.loads(name.read_text())

    (args.output_dir / "arguments.json").write_text(
        json.dumps(
            {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2
        )
        + "\n"
    )
    try:
        status(pose_run=str(args.pose_run))
        deadline = time.monotonic() + args.wait_timeout_hours * 3600
        while not (args.pose_run / "completion.json").exists():
            if stopping:
                raise InterruptedError("Pipeline interrupted while waiting for pose training")
            if (args.pose_launch / "exit_code").exists():
                raise RuntimeError("Pose process exited without completion.json")
            if time.monotonic() > deadline:
                raise TimeoutError("Pose training did not finish within the configured wait timeout")
            time.sleep(15)
        completion = read(args.pose_run / "completion.json")
        if not completion["completed"]:
            raise RuntimeError(f"Pose training did not complete successfully: {completion}")
        pose_checkpoint = args.pose_run / "best.pth"
        pose_evaluation = args.output_dir / "pose_evaluation.json"
        run(
            "evaluate_pose",
            [
                "bash",
                "scripts/run_sim6.sh",
                "scripts/pose_estimator/evaluate_pose_dataset.py",
                "--checkpoint",
                str(pose_checkpoint),
                "--dataset",
                str(args.validation_dataset),
                "--output",
                str(pose_evaluation),
            ],
        )
        evaluation = read(pose_evaluation)
        if not evaluation["completed"] or evaluation["all"]["count"] < 1:
            raise RuntimeError("Pose evaluation is incomplete")
        if not all(math.isfinite(value) for value in evaluation["all"]["mean"].values()):
            raise RuntimeError("Pose evaluation produced non-finite metrics")
        teacher_sha = hashlib.sha256(args.teacher_checkpoint.read_bytes()).hexdigest()
        if evaluation["teacher_sha256"] != teacher_sha:
            raise RuntimeError("Pose dataset and distillation teacher checkpoints differ")
        # Test the frozen teacher with the same noisy proprioception used for labels.
        run(
            "teacher_baseline",
            [
                "bash",
                "scripts/distill_revo3_5090.sh",
                "--evaluate",
                "--teacher_only",
                "--teacher_checkpoint",
                str(args.teacher_checkpoint),
                "--num_envs",
                "64",
                "--num_steps",
                "1200",
                "--seed",
                "53",
                "--output_dir",
                str(args.output_dir / "teacher_baseline"),
            ],
        )
        student_dir = args.output_dir / "student"
        run(
            "distillation",
            [
                "bash",
                "scripts/distill_revo3_5090.sh",
                "--teacher_checkpoint",
                str(args.teacher_checkpoint),
                "--num_envs",
                str(args.num_envs),
                "--max_iterations",
                str(args.max_iterations),
                "--epochs",
                "32",
                "--rollout_steps",
                "45",
                "--save_interval",
                "50",
                "--seed",
                "42",
                "--output_dir",
                str(student_dir),
            ],
        )
        student_completion = read(student_dir / "completion.json")
        if not student_completion["completed"]:
            raise RuntimeError("Student training was interrupted; evaluations were not started")
        checkpoint = student_completion["checkpoint"]
        for name, ablation in (("student_eval_noisy", []), ("student_eval_clean", ["--clean_observations"])):
            run(
                name,
                [
                    "bash",
                    "scripts/distill_revo3_5090.sh",
                    "--evaluate",
                    "--student_checkpoint",
                    checkpoint,
                    "--num_envs",
                    "64",
                    "--num_steps",
                    "1200",
                    "--seed",
                    "53",
                    "--output_dir",
                    str(args.output_dir / name),
                    *ablation,
                ],
            )
        run(
            "export",
            [
                "bash",
                "scripts/run_sim6.sh",
                "scripts/rsl_rl/export_student.py",
                "--checkpoint",
                checkpoint,
                "--output_dir",
                str(student_dir / "exported"),
            ],
        )
        stage = "completed"
        status(checkpoint=checkpoint, pose_evaluation=str(pose_evaluation))
    except BaseException as error:
        status(failed=True, error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
