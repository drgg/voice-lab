"""Super-resolve experiment outputs to 48 kHz with AudioSR (run in envs/audiosr).

Seed-VC V2 and IndexTTS-2.5 render at 22.05 kHz (nothing above ~11 kHz), which is what makes them
sound dull next to the 48 kHz game originals. AudioSR's speech model regenerates the missing high
band. Results go to exp/<group>_48k/ so they can be compared with the 22 kHz version; afterwards run
postprocess.py on the new groups for fades / loudness matching.
A line whose 48 kHz file is newer than its source is skipped (re-runs only redo what changed).

Usage:  envs/audiosr/Scripts/python scripts/superres.py --groups A2_seedvc_emo,D2_indextts_emo [--force]
"""
import argparse
import json
import math
import os
import tempfile
import time

import numpy as np
import soundfile as sf

from vl_paths import EXP

SR_OUT = 48000
UNIT = 5.12  # AudioSR works on 5.12 s blocks


def pieces(wav, sr, max_len=10.24, search=1.5):
    """Cut long audio at its quietest 20 ms near every max_len seconds: AudioSR degrades past 10.24 s
    (and a 67 s monologue would not fit in VRAM in one go)."""
    n, step, hop = len(wav), int(max_len * sr), int(0.02 * sr)
    cuts = [0]
    while n - cuts[-1] > step:
        lo, hi = cuts[-1] + int((max_len - search) * sr), cuts[-1] + step
        frames = wav[lo:hi][: (hi - lo) // hop * hop].reshape(-1, hop)
        cuts.append(lo + int(np.argmin((frames ** 2).mean(axis=1))) * hop + hop // 2)
    return list(zip(cuts, cuts[1:] + [n]))


def upsample(model, wav, sr, a):
    """AudioSR pads its input to 5.12 s blocks with float arithmetic, so some lengths (1.1 s at 40 kHz)
    end up one frame short and crash its UNet; pad to an exact block length first, trim afterwards."""
    out = []
    for s, e in pieces(wav, sr):
        block = int(round(UNIT * sr))
        x = np.zeros(math.ceil((e - s) / block) * block, dtype=np.float32)
        x[:e - s] = wav[s:e]
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(tmp, x, sr, subtype="FLOAT")
            y = super_resolution(model, tmp, seed=a.seed, guidance_scale=a.guidance,
                                 ddim_steps=a.steps, latent_t_per_second=12.8)
        finally:
            os.remove(tmp)
        out.append(np.asarray(y, dtype=np.float32).reshape(-1)[:int(round((e - s) / sr * SR_OUT))])
    for k in range(1, len(out)):  # 5 ms fades where long audio was cut (cuts sit in its quietest spots)
        n = min(int(0.005 * SR_OUT), len(out[k]) // 2, len(out[k - 1]) // 2)
        out[k - 1][len(out[k - 1]) - n:] *= np.linspace(1, 0, n)
        out[k][:n] *= np.linspace(0, 1, n)
    return np.concatenate(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    ap.add_argument("--steps", type=int, default=50, help="DDIM steps")
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="redo lines that are already up to date")
    ap.add_argument("--plain", action="store_true",
                    help="only resample to 48 kHz: for RVC output (40 kHz, already full band) of grunts and "
                         "breaths, where AudioSR - a speech model - invents detail")
    a = ap.parse_args()

    if a.plain:
        from math import gcd

        from scipy.signal import resample_poly
    else:
        from audiosr import build_model, super_resolution  # module-level names: upsample() uses super_resolution

        # peak commit ~15.5 GB while the 5.75 GB checkpoint loads: keep the pagefile off a nearly full C:
        model = build_model(model_name="speech", device="cuda")
    for g in a.groups.split(","):
        src_dir, out_dir = EXP / g, EXP / f"{g}_48k"
        out_dir.mkdir(parents=True, exist_ok=True)
        log = []
        for src in sorted(src_dir.glob("L*.wav")):
            dst = out_dir / src.name
            if not a.force and dst.exists() and dst.stat().st_mtime > src.stat().st_mtime:
                continue
            t = time.time()
            x, sr = sf.read(str(src), dtype="float32", always_2d=True)
            if a.plain:  # 40 kHz -> 48 kHz is an exact 6/5 polyphase resample
                g = gcd(SR_OUT, sr)
                wav = resample_poly(x.mean(axis=1), SR_OUT // g, sr // g).astype(np.float32)
            else:
                wav = upsample(model, x.mean(axis=1), sr, a)
            peak = float(np.abs(wav).max())
            if peak > 0.97:
                wav *= 0.97 / peak
            sf.write(dst, wav, SR_OUT, subtype="PCM_16")
            log.append({"id": src.stem, "seconds": round(time.time() - t, 1)})
            print(f"[{g}_48k] {src.stem} {log[-1]['seconds']}s")
        if not log:
            continue
        (out_dir / "run.json").write_text(json.dumps(
            {"params": {"model": "resample" if a.plain else "speech", "steps": a.steps, "guidance": a.guidance,
                        "seed": a.seed},
             "lines": log}, indent=2), encoding="utf-8")
