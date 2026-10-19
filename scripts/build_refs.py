"""Assemble target reference audio from the ranked clips (run after `prep_voice.py --stage rank`).

Outputs in data/<voice>/refs/:
  ref_25s.wav   best clips joined up to ~25 s  (Seed-VC target; Seed-VC clips refs at 25 s)
  ref_12s.wav   best clips joined to 8-12 s   (IndexTTS speaker prompt, Vevo2 timbre ref)
  style_ref.wav single clean clip 4-10 s with its text (Vevo2 style ref needs a transcript)
  abx_01..05    other clean target clips used as "this is the target" references in the listening test
  refs.json     which clips went where

Usage (tools env):  python scripts/build_refs.py [--min-score 0.6]
"""
import argparse
import json

import numpy as np
import soundfile as sf

from vl_common import TARGET, is_target_clip, read_csv, read_mono

REFS = TARGET / "refs"
SR = 44100
GAP = np.zeros(int(0.25 * SR), dtype=np.float32)


def clean_pool(min_score):
    rows = read_csv(TARGET / "slices_ranked.csv")
    pool = [
        r for r in rows
        if r["anchor"] != "0" and float(r["score"]) >= min_score and r["radio"] != "True"
        and float(r["bg_ratio_db"]) < -10 and 2.0 <= float(r["duration"]) <= 10.0
    ]
    if any(r.get("p_target") for r in pool):  # after `classify`: also the speaker classifier and speech habits
        pool = [r for r in pool if is_target_clip(r, min_score=min_score, max_bg=-10, min_dur=2.0, max_dur=10.0)]
    if r_neg := [r for r in pool if r.get("neg_score")]:
        # drop clips that look more like the "not the target" marks than like the target
        bad = {r["id"] for r in r_neg if float(r["neg_score"]) > float(r["score"])}
        pool = [r for r in pool if r["id"] not in bad]
    return pool


def load(cid):
    clean = TARGET / "slices_clean" / f"{cid}.wav"  # de-reverbed / de-noised by clean_refs.py, when present
    wav, _ = read_mono(clean if clean.exists() else TARGET / "slices" / f"{cid}.wav", sr=SR)
    return wav / (np.abs(wav).max() + 1e-9) * 0.9


def join(ids):
    parts = []
    for cid in ids:
        parts += [load(cid), GAP]
    return np.concatenate(parts[:-1]) if parts else np.zeros(0, np.float32)


def take_until(pool, lo, hi, used):
    ids, total = [], 0.0
    for r in pool:
        d = float(r["duration"])
        if r["id"] in used or total + d > hi:
            continue
        ids.append(r["id"])
        total += d + 0.25
        if total >= lo:
            break
    return ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=0.6)
    a = ap.parse_args()
    pool = clean_pool(a.min_score)
    if len(pool) < 8:
        raise SystemExit(f"only {len(pool)} clean clips above score {a.min_score}; lower --min-score or mark more anchors")
    REFS.mkdir(parents=True, exist_ok=True)
    texts = {r["id"]: r.get("text", "") for r in pool}

    style = next(r for r in pool if 4.0 <= float(r["duration"]) <= 10.0 and texts[r["id"]])
    used = {style["id"]}
    ids12 = take_until(pool, 8, 12, used)
    ids25 = take_until(pool, 20, 25, set())
    abx = [r["id"] for r in pool if r["id"] not in used | set(ids12)][:5]

    sf.write(REFS / "style_ref.wav", load(style["id"]), SR, subtype="PCM_16")
    (REFS / "style_ref.txt").write_text(texts[style["id"]], encoding="utf-8")
    sf.write(REFS / "ref_12s.wav", join(ids12), SR, subtype="PCM_16")
    sf.write(REFS / "ref_25s.wav", join(ids25), SR, subtype="PCM_16")
    for i, cid in enumerate(abx, 1):
        sf.write(REFS / f"abx_{i:02d}.wav", load(cid), SR, subtype="PCM_16")
    meta = {"style_ref": style["id"], "style_text": texts[style["id"]], "ref_12s": ids12,
            "ref_25s": ids25, "abx": abx, "min_score": a.min_score, "pool_size": len(pool)}
    (REFS / "refs.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
