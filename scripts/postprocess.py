"""Post-process experiment outputs in place (tools env); originals are kept in exp/<group>/_raw/.

  de-click   single-sample spikes far above the local level are replaced by interpolation
  fades      5 ms fade in/out so lines start and end without a click
  loudness   integrated loudness matched to the original line (consistent in game),
             with the peak kept under -1 dBFS
Always processes from _raw/, so it can be re-run safely; files already processed are skipped.

Usage:  python scripts/postprocess.py --groups A2_seedvc_emo,D2_indextts_emo
"""
import argparse
import json
import shutil

import numpy as np
import pyloudnorm
import soundfile as sf

from vl_common import EXP, LINES, read_mono

CEILING = 10 ** (-1 / 20)  # -1 dBFS


def declick(x, thresh=0.25, ratio=8.0):
    d = np.abs(np.diff(x))
    local = np.convolve(np.abs(x), np.ones(64) / 64, mode="same")[1:] + 1e-4
    spikes = np.where((d > thresh) & (d > ratio * local))[0] + 1
    y = x.copy()
    for i in spikes:
        lo, hi = max(i - 2, 0), min(i + 3, len(y))
        y[lo:hi] = np.linspace(y[lo - 1] if lo > 0 else 0.0, y[hi] if hi < len(y) else 0.0, hi - lo)
    return y, len(spikes)


def fade(x, sr, ms=5):
    n = min(int(sr * ms / 1000), len(x) // 2)
    ramp = np.linspace(0.0, 1.0, n, dtype=x.dtype)
    x[:n] *= ramp
    x[len(x) - n:] *= ramp[::-1]
    return x


def match_loudness(x, sr, ref, ref_sr):
    try:
        target = pyloudnorm.Meter(ref_sr).integrated_loudness(ref)
        current = pyloudnorm.Meter(sr).integrated_loudness(x)
        if np.isfinite(target) and np.isfinite(current):
            x = x * 10 ** ((target - current) / 20)
    except ValueError:  # shorter than one 0.4 s loudness block (grunts): match the RMS level instead
        x = x * (np.sqrt(np.mean(ref ** 2)) + 1e-9) / (np.sqrt(np.mean(x ** 2)) + 1e-9)
    peak = np.abs(x).max()
    return x * (CEILING / peak) if peak > CEILING else x


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    a = ap.parse_args()
    for g in a.groups.split(","):
        folder, raw = EXP / g, EXP / g / "_raw"
        if not any(folder.glob("L*.wav")):
            print(f"{g}: no outputs to post-process - skipped")  # e.g. a set without non-verbal lines
            continue
        raw.mkdir(exist_ok=True)
        manifest_p = raw / "processed.json"
        manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else {}
        total = 0
        for out in sorted(folder.glob("L*.wav")):
            # a file whose mtime differs from what we last wrote is fresh runner output -> new raw;
            # one we already processed is left alone, so the 48 kHz copy made from it stays up to date
            if manifest.get(out.name) == out.stat().st_mtime_ns and (raw / out.name).exists():
                continue
            shutil.copy2(out, raw / out.name)
            x, sr = read_mono(raw / out.name)
            ref, ref_sr = read_mono(LINES / out.name)
            x, n = declick(x)
            x = match_loudness(fade(x, sr), sr, ref, ref_sr)
            sf.write(out, x, sr, subtype="PCM_16")
            manifest[out.name] = out.stat().st_mtime_ns
            total += n
        manifest_p.write_text(json.dumps(manifest, indent=1))
        print(f"{g}: post-processed, {total} clicks repaired")
