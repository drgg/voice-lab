"""Does a clip of the target voice (VL_VOICE) hold someone else's words too? (tools env)

A clip is cut at pauses, so a quick exchange can end up in one clip: test line T28 of the voice's set carried another character's
reply after hers (the user heard it, 2026-10-15), and the clip-level ranking had passed it. Per clip:
  windows   ECAPA embeddings of 1.2 s windows (0.4 s hop) over the voiced part: cosine to her training centroid
            and to the clip's own mean; a run of windows far below the rest is someone else
  low voice voiced stretches below 160 Hz lasting >= 0.4 s (her floor is far above: a male voice)
Writes data/<voice>/purity.csv (all clips of train.csv, or --ids) and prints the flagged ones.

  set VL_VOICE=myvoice & python scripts/clip_purity.py [--ids id1,id2] [--keys T28,T18]
"""
import argparse
import random

import numpy as np

import prosody
from emotion_match import clip_path
from prep_voice import ecapa, embed
from vl_common import TARGET, read_csv, read_mono, write_csv

WIN, HOP = 1.2, 0.4


def windows(model, path):
    import torch

    wav, _ = read_mono(path, sr=16000)
    n, h = int(WIN * 16000), int(HOP * 16000)
    frames = wav[: len(wav) // 160 * 160].reshape(-1, 160)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-9)
    loud = db > np.percentile(db, 90) - 25  # skip windows that are mostly pause
    out = []
    for s in range(0, max(1, len(wav) - n + 1), h):
        if loud[s // 160:(s + n) // 160].mean() < 0.5:
            continue
        with torch.no_grad():
            e = model.encode_batch(torch.from_numpy(wav[s:s + n])[None].cuda()).squeeze().cpu().numpy()
        out.append((s / 16000, e / (np.linalg.norm(e) + 1e-9)))
    return out


def low_voice(path, floor=160.0, min_s=0.4):
    wav, sr = read_mono(path)
    st, _ = prosody.semitones(wav, sr)
    hz = 100 * 2 ** (st / 12)
    low = np.isfinite(hz) & (hz < floor)
    best = run = 0
    for v in low:
        run = run + 1 if v else 0
        best = max(best, run)
    return best * 0.01 if best * 0.01 >= min_s else 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="")
    ap.add_argument("--keys", default="", help="test-set keys (tts/testset.csv)")
    a = ap.parse_args()
    train = read_csv(TARGET / "train.csv")
    ids = [x for x in a.ids.split(",") if x]
    if a.keys:
        tk = {r["key"]: r["id"] for r in read_csv(TARGET / "tts" / "testset.csv")}
        ids += [tk[k] for k in a.keys.split(",")]
    if not ids:
        ids = [r["id"] for r in train]
    model = ecapa()
    cen = np.mean([embed(model, clip_path(r["id"])) for r in random.Random(1).sample(train, min(300, len(train)))], 0)
    cen /= np.linalg.norm(cen)
    rows = []
    for k, cid in enumerate(ids, 1):
        p = clip_path(cid)
        ws = windows(model, p)
        if ws:
            sims = np.array([float(e @ cen) for _, e in ws])
            mean = np.mean([e for _, e in ws], 0)
            own = np.array([float(e @ mean / np.linalg.norm(mean)) for _, e in ws])
            worst = int(np.argmin(sims))
            gap = float(np.median(sims) - sims.min())
        else:
            sims, own, worst, gap = np.array([np.nan]), np.array([np.nan]), 0, 0.0
        lv = low_voice(p)
        flag = []
        if len(ws) >= 3 and sims.min() < 0.30 and gap > 0.30:
            flag.append("other voice?")
        if lv:
            flag.append("low voice")
        rows.append({"id": cid, "windows": len(ws), "sim_median": round(float(np.nanmedian(sims)), 3),
                     "sim_min": round(float(np.nanmin(sims)), 3), "gap": round(gap, 3),
                     "worst_at_s": round(ws[worst][0], 1) if ws else "", "own_min": round(float(np.nanmin(own)), 3),
                     "low_voice_s": round(lv, 2), "flag": "; ".join(flag)})
        if k % 100 == 0:
            print(f"  {k}/{len(ids)}", flush=True)
    out = TARGET / ("purity.csv" if not (a.ids or a.keys) else "purity_check.csv")
    write_csv(out, rows)
    for r in rows:
        if r["flag"] or a.keys or a.ids:
            print(f"{r['id'][-34:]}  windows {r['windows']}  sim median {r['sim_median']} min {r['sim_min']} "
                  f"(at {r['worst_at_s']} s) gap {r['gap']}  low voice {r['low_voice_s']} s  {r['flag']}")
    print(f"{sum(1 for r in rows if r['flag'])} of {len(rows)} flagged -> {out}")
