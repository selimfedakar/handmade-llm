"""Chapter 00 — measure the machine before you trust it.

Every later chapter asks "will this fit?", and the honest answer depends on
your chip and your memory. So we start by measuring, not guessing.

    python 00_setup/check_mac.py

It prints what you have, what it can do, and roughly how large a model you can
train before the system starts swapping.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time

GIB = 1024**3


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def machine_info() -> dict:
    """What chip, how much memory, how many GPU cores."""
    total_bytes = _sysctl("hw.memsize")
    return {
        "os": f"{platform.system()} {platform.release()}",
        "chip": _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown",
        "cpu_cores": _sysctl("hw.ncpu"),
        "gpu_cores": _sysctl("hw.perflevel0.gpucorecount"),
        "memory_gib": int(total_bytes) / GIB if total_bytes else None,
        "python": platform.python_version(),
    }


def matmul_throughput(size: int = 2048, dtype_name: str = "float16") -> dict:
    """Time a square matmul to get a rough ceiling on this machine's GPU.

    This is not a marketing benchmark. It is a sanity number: if chapter 03
    feels slow later, come back here and check that nothing has changed.
    """
    import mlx.core as mx

    dtype = getattr(mx, dtype_name)
    a = mx.random.normal((size, size)).astype(dtype)
    b = mx.random.normal((size, size)).astype(dtype)

    # MLX is lazy: nothing runs until you ask for the result. Without the
    # eval() calls below you would be timing graph construction, which is
    # fast and meaningless.
    mx.eval(a, b)
    for _ in range(3):  # warm up the kernels and the clocks
        mx.eval(a @ b)

    runs = 10
    start = time.perf_counter()
    for _ in range(runs):
        mx.eval(a @ b)
    seconds = (time.perf_counter() - start) / runs

    flops = 2 * size**3  # one multiply and one add per inner-product step
    return {"size": size, "dtype": dtype_name, "seconds": seconds, "tflops": flops / seconds / 1e12}


def memory_budget(total_gib: float) -> dict:
    """Split total memory into what you can actually spend on training.

    macOS and your open apps need their share. What is left has to hold
    parameters, gradients, the Adam optimizer state, and activations — and in
    mixed-precision Adam training those come out near 16 bytes per parameter
    once activations are counted in.
    """
    reserved = max(4.0, total_gib * 0.25)  # OS, browser, editor, the usual
    usable = max(0.0, total_gib - reserved)
    bytes_per_param = 16
    params = usable * GIB / bytes_per_param
    return {
        "total_gib": total_gib,
        "reserved_gib": reserved,
        "usable_gib": usable,
        "trainable_params_millions": params / 1e6,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=2048, help="matmul size for the throughput test")
    parser.add_argument("--json", action="store_true", help="print machine-readable output")
    args = parser.parse_args()

    info = machine_info()

    if platform.system() != "Darwin":
        print("This repository targets Apple Silicon. You are on:", info["os"], file=sys.stderr)
        return 1

    try:
        import mlx.core as mx
    except ImportError:
        print("MLX is not installed. Run:  pip install -r requirements.txt", file=sys.stderr)
        return 1

    bench = matmul_throughput(args.size)
    budget = memory_budget(info["memory_gib"]) if info["memory_gib"] else None

    if args.json:
        import json

        print(json.dumps({"machine": info, "matmul": bench, "budget": budget}, indent=2))
        return 0

    print("Machine")
    print(f"  chip           {info['chip']}")
    print(f"  cpu cores      {info['cpu_cores']}")
    if info["gpu_cores"]:
        print(f"  gpu cores      {info['gpu_cores']}")
    print(f"  memory         {info['memory_gib']:.0f} GiB unified")
    print(f"  python / mlx   {info['python']} / {mx.__version__}")

    print("\nThroughput")
    print(f"  {bench['size']}x{bench['size']} {bench['dtype']} matmul")
    print(f"  {bench['seconds'] * 1000:.1f} ms per run   ->   {bench['tflops']:.2f} TFLOP/s")

    if budget:
        print("\nMemory budget for training")
        print(f"  reserved for the system   {budget['reserved_gib']:.1f} GiB")
        print(f"  yours to spend            {budget['usable_gib']:.1f} GiB")
        print(
            f"  roughly                   {budget['trainable_params_millions']:.0f}M parameters"
            "  (mixed-precision Adam, activations included)"
        )
        print("\n  That number is a ceiling, not a target. Chapter 04 measures the real thing.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
