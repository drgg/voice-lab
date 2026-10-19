"""Run Seed-VC's train_v2.py with guard rails (envs/seedvc).

On this machine (16 GB VRAM / 16 GB RAM) the V2 fine-tune overflowed VRAM and the Windows driver
silently spilled into system RAM (40+ GB private memory within a minute, pagefile growth filled C:).
This wrapper
  - caps PyTorch's CUDA allocator at a fraction of VRAM, so an overflow raises OOM instead of spilling
  - runs a watchdog thread that kills the process once its private memory passes --max-ram-gb
then hands the remaining arguments to train_v2.py unchanged.

Usage:  envs/seedvc/Scripts/python scripts/train_seedvc_safe.py [--vram-frac 0.85] [--max-ram-gb 12] -- <train_v2 args>
"""
import os
import runpy
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] / "repos" / "seed-vc"


def guard(limit_gb):
    import psutil

    proc = psutil.Process()
    while True:
        rss = proc.memory_info().private / 2**30 if hasattr(proc.memory_info(), "private") else proc.memory_info().rss / 2**30
        if rss > limit_gb:
            print(f"[guard] private memory {rss:.1f} GB > {limit_gb} GB - aborting to protect the system", flush=True)
            os._exit(3)
        time.sleep(2)


if __name__ == "__main__":
    args = sys.argv[1:]
    split = args.index("--") if "--" in args else len(args)
    own, train_args = args[:split], args[split + 1:]
    vram_frac = float(own[own.index("--vram-frac") + 1]) if "--vram-frac" in own else 0.85
    max_ram = float(own[own.index("--max-ram-gb") + 1]) if "--max-ram-gb" in own else 12.0

    threading.Thread(target=guard, args=(max_ram,), daemon=True).start()
    import torch

    torch.cuda.set_per_process_memory_fraction(vram_frac)

    # 173 clips at batch 2 leave a final batch of 1, which breaks the CFM style encoder's BatchNorm
    class DropLastLoader(torch.utils.data.DataLoader):
        def __init__(self, *a, **kw):
            if (kw.get("batch_size") or 1) > 1 and "batch_sampler" not in kw:
                kw.setdefault("drop_last", True)
            super().__init__(*a, **kw)

    torch.utils.data.DataLoader = DropLastLoader
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    sys.argv = ["train_v2.py", *train_args]
    runpy.run_path("train_v2.py", run_name="__main__")
