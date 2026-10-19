"""Export the the target training set for fine-tuning (Seed-VC V2 train_v2.py / RVC) from the ranked clips.

Run `prep_voice.py --stage rank ...` and `--stage classify` first, then clean_refs.py. Keeps only
clips that pass vl_common.is_target_clip (classifier + similarity + no other characters' speech
markers + low background), using the de-reverbed / de-noised copies when present; writes them
(peak-normalised) to data/<voice>/train/ plus train.csv with the total duration.

Usage (tools env):  python scripts/build_trainset.py [--min-p 0.7] [--min-score 0.6]
"""
import argparse
import shutil

import numpy as np
import soundfile as sf

from vl_common import TARGET, is_target_clip, read_csv, read_mono, write_csv

TRAIN = TARGET / "train"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-p", type=float, default=0.7)
    ap.add_argument("--min-score", type=float, default=0.6)
    a = ap.parse_args()

    rows = [r for r in read_csv(TARGET / "slices_ranked.csv") if is_target_clip(r, a.min_p, a.min_score)]
    # two segmentations of the same speech (prep_voice add_cuts_from) overlap: the longest clip of a span wins,
    # others only if under 30 % of them is already taken
    taken, keep = {}, []
    for r in sorted(rows, key=lambda r: -float(r["duration"])):
        s, e = float(r["start"]), float(r["end"])
        iv = taken.setdefault(r["episode"], [])
        if sum(max(0.0, min(e, e2) - max(s, s2)) for s2, e2 in iv) < 0.3 * (e - s):
            iv.append((s, e))
            keep.append(r)
    if len(keep) < len(rows):
        print(f"{len(rows) - len(keep)} overlapping clips dropped (same speech cut twice)")
    rows = sorted(keep, key=lambda r: r["id"])
    if TRAIN.exists():
        shutil.rmtree(TRAIN)
    TRAIN.mkdir(parents=True)
    out = []
    for i, r in enumerate(rows):
        clean = TARGET / "slices_clean" / f"{r['id']}.wav"  # de-reverbed / de-noised by clean_refs.py
        wav, sr = read_mono(clean if clean.exists() else TARGET / "slices" / f"{r['id']}.wav")
        wav = wav / (np.abs(wav).max() + 1e-9) * 0.9
        name = f"clip_{i:04d}.wav"  # short ASCII names: training code paths stay simple
        sf.write(TRAIN / name, wav, sr, subtype="PCM_16")
        out.append({"file": name, "id": r["id"], "duration": r["duration"], "score": r["score"],
                    "p_target": r["p_target"], "cleaned": clean.exists(), "text": r.get("text", "")})
    write_csv(TARGET / "train.csv", out)
    total = sum(float(r["duration"]) for r in out)
    eps = sorted({r["id"].split("_-_")[1][:2] for r in out if "_-_" in r["id"]})
    print(f"{len(out)} clips, {total / 60:.1f} min from episodes {eps} "
          f"({sum(r['cleaned'] for r in out)} cleaned) -> {TRAIN}")
