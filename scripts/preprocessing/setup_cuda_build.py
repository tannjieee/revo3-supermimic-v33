"""Expose CUDA 13 wheel's runtime library to the native extension linker."""

from pathlib import Path
import sys


def main():
    prefix = Path(sys.prefix)
    cuda = prefix / "lib/python3.12/site-packages/nvidia/cu13"
    if not (cuda / "bin/nvcc").is_file():
        raise FileNotFoundError("Install cuda-toolkit[nvcc,cudart,cccl]==13.0.2 in this environment first")
    runtime = cuda / "lib/libcudart.so.13"
    if not runtime.is_file():
        raise FileNotFoundError(runtime)
    linker_dir = prefix / "lib/cuda-build"
    linker_dir.mkdir(parents=True, exist_ok=True)
    link = linker_dir / "libcudart.so"
    if link.exists() or link.is_symlink():
        if link.resolve() != runtime.resolve():
            raise RuntimeError(f"Existing CUDA linker alias has a different target: {link}")
    else:
        link.symlink_to(runtime)
    print(f"CUDA_BUILD_READY: {link} -> {runtime}")


if __name__ == "__main__":
    main()
