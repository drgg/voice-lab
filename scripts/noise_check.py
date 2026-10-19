"""Diagnose noise in outputs and references (tools env).

Per file: sample rate, peak / clipped share, noise floor (10th percentile of 20 ms frame RMS),
hiss in pauses (share of >6 kHz energy in the quietest frames), and clicks (single-sample jumps far
above the local signal). Prints per-group medians; writes eval/noise_check.csv.
Usage:  python scripts/noise_check.py
"""
import numpy as np
import soundfile as sf

from vl_common import EVAL, EXP, LINES, METHODS, TARGET, write_csv


def analyse(path):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = wav.mean(axis=1)
    hop = int(0.02 * sr)
    frames = x[: len(x) // hop * hop].reshape(-1, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1)) + 1e-9
    db = 20 * np.log10(rms)
    floor = float(np.percentile(db, 10))
    quiet = frames[db <= np.percentile(db, 15)]
    if len(quiet):
        spec = np.abs(np.fft.rfft(quiet * np.hanning(hop), axis=1)) ** 2
        f = np.fft.rfftfreq(hop, 1 / sr)
        hiss = float(spec[:, f >= 6000].sum() / (spec.sum() + 1e-12))
    else:
        hiss = float("nan")
    # high band (11-20 kHz) share in the loudest frames: ~0 for 22.05 kHz renders, restored by super-res
    loud = frames[db >= np.percentile(db, 70)]
    if sr > 24000 and len(loud):
        spec = np.abs(np.fft.rfft(loud * np.hanning(hop), axis=1)) ** 2
        f = np.fft.rfftfreq(hop, 1 / sr)
        hf_band = float(spec[:, (f >= 11000) & (f < 20000)].sum() / (spec.sum() + 1e-12))
    else:
        hf_band = 0.0
    d = np.abs(np.diff(x))
    local = np.convolve(np.abs(x), np.ones(64) / 64, mode="same")[1:] + 1e-4
    clicks = int(((d > 0.25) & (d > 8 * local)).sum())
    return {
        "sr": sr, "peak": round(float(np.abs(x).max()), 3),
        "clip_pct": round(100 * float((np.abs(x) >= 0.999).mean()), 3),
        "floor_db": round(floor, 1), "dyn_db": round(float(np.percentile(db, 95)) - floor, 1),
        "hiss_pause": round(hiss, 3), "hf_band": round(hf_band, 4), "clicks": clicks,
    }


if __name__ == "__main__":
    groups = [("source_orig", sorted(LINES.glob("L*.wav")))]
    groups += [(m, sorted((EXP / m).glob("L*.wav"))) for m in METHODS if (EXP / m).exists()]
    groups += [("ref_round1", [TARGET / "refs/ref_25s.wav", TARGET / "refs/ref_12s.wav"]),
               ("ref_emo_round2", sorted((TARGET / "refs/emo").glob("L*.wav")))]
    rows = []
    for g, files in groups:
        for p in files:
            rows.append({"group": g, "file": p.name, **analyse(p)})
    write_csv(EVAL / "noise_check.csv", rows)
    print(f"{'group':22s} {'n':>3s} {'sr':>6s} {'peak':>5s} {'clip%':>6s} {'floor':>6s} {'dyn':>5s} "
          f"{'hiss':>5s} {'hf11k':>6s} {'clicks':>6s}")
    for g, _ in groups:
        rs = [r for r in rows if r["group"] == g]
        if not rs:
            continue
        med = lambda k: float(np.median([r[k] for r in rs]))  # noqa: E731
        print(f"{g:22s} {len(rs):3d} {rs[0]['sr']:6d} {med('peak'):5.2f} {med('clip_pct'):6.3f} "
              f"{med('floor_db'):6.1f} {med('dyn_db'):5.1f} {med('hiss_pause'):5.3f} {med('hf_band'):6.4f} "
              f"{sum(r['clicks'] for r in rs):6d}")
