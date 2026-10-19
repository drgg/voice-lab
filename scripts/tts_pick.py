"""Pick the best of several takes per line from a text-to-speech system (tools env; VL_VOICE).

  python scripts/tts_pick.py <takes dir> --list <speak csv> --out <dir>

<takes dir> holds <key>_t<k>.wav (gsv.py speak --takes N); the speak csv gives each line's text and style reference.
The fine-tuned model drifts toward her average manner (myvoice_gsv1: emotion2vec heard "surprised" in 27 of 60 lines
against 19 in the real clips, angry 8 against 14) and followed its reference's category only half the time, so the
take closest to the reference wins. Only what the system was given is used - never the clip being re-dubbed:
  emotion  cosine of the emotion2vec embeddings (her bank's mean removed) to the reference, + --w-cat (0.5) for
           the same top category
  arousal  distance of the arousal cues to the reference's (loudness left out, as in tts_eval.py)
  timbre   ECAPA cosine to her training centroid
  words    a take whose Whisper transcript misses more than 10 % of the text loses 3 per 0.1 above that
  ending   (--w-ending) how she ends lines with the same last characters: the take's last 0.5 s of voiced pitch
           (relative to its median, 20 points) against her training clips ending alike (the longest common
           suffix of 3/2/1 characters with >= 4 examples, a written ? kept apart), mean of the 5 closest.
           Added 2026-10-15: the takes held endings her way in 32 of 32 lines with a clear habit, the pick in 20.
  quality  (--w-quality) SQUIM's PESQ estimate: prompt-free takes (gsv.py --no-prompt-text) are livelier but
           some come out rough (PESQ median 3.75 against 3.98)
Each term is standardized over the line's takes. Writes <out>/<key>.wav and pick.csv.
"""
import argparse
import random
import shutil
from pathlib import Path

import numpy as np

from arousal_features import features as arousal_raw
from emotion_match import AROUSAL, clip_path
from tts_eval import cer, emotions
from vl_common import TARGET, read_csv, transcribe, whisper_model, write_csv


def final_shape(path, n=20):
    """The last 0.5 s of voiced pitch, relative to the clip's median, resampled to n points (None if too short)."""
    import prosody
    from vl_common import read_mono

    st, _ = prosody.semitones(*read_mono(path))
    v = np.where(np.isfinite(st))[0]
    if len(v) < 15:
        return None
    tail = st[v[-50:]] - np.nanmedian(st)
    return np.interp(np.linspace(0, len(tail) - 1, n), np.arange(len(tail)), tail)


def ending_key(text):
    import re

    t = re.sub(r"[。．、\s」』）)…‥!！]+$", "", text.strip())
    q = bool(re.search(r"[?？]$", t))
    return re.sub(r"[?？]+$", "", t), q


def ending_bank(train):
    """Her training clips' endings (text without final punctuation, question?, shape), cached by clip ids."""
    cache = TARGET / "tts" / "ending_shapes.npz"
    ids = [r["id"] for r in train]
    if cache.exists():
        zc = np.load(cache, allow_pickle=False)
        if list(zc["ids"]) == ids:
            return [(str(tx), bool(q), s) for tx, q, s in zip(zc["texts"], zc["q"], zc["shapes"])]
    bank, keep = [], []
    for r in train:
        s = final_shape(clip_path(r["id"]))
        tx, q = ending_key(r["text"])
        bank.append((tx, q, s if s is not None else np.full(20, np.nan)))
    np.savez(cache, ids=np.array(ids), texts=np.array([b[0] for b in bank]), q=np.array([b[1] for b in bank]),
             shapes=np.stack([b[2] for b in bank]))
    return bank


def ending_score(shape, text, bank):
    """Minus the mean distance to her 5 closest endings that end like this text (see the docstring)."""
    if shape is None:
        return -9.0
    tx, q = ending_key(text)
    pool = []
    for n in (3, 2, 1):
        if len(tx) >= n:
            pool = [s for t2, q2, s in bank if q2 == q and t2.endswith(tx[-n:]) and np.isfinite(s).all()]
            if len(pool) >= 4:
                break
    if len(pool) < 4:
        pool = [s for t2, q2, s in bank if q2 == q and np.isfinite(s).all()]
    d = np.sort([float(np.sqrt(np.mean((shape - s) ** 2))) for s in pool])[:5]
    return -float(d.mean())


def z(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9) if len(x) > 1 else np.zeros_like(x)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("takes_dir")
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w-cat", type=float, default=0.5, help="bonus for the reference's emotion2vec category")
    ap.add_argument("--w-timbre", type=float, default=1.0,
                    help="weight of the timbre term (2 after the user's ratings followed timbre most, 2026-10-15)")
    ap.add_argument("--w-ending", type=float, default=0.0, help="weight of her ending habit (see the docstring)")
    ap.add_argument("--w-quality", type=float, default=0.0, help="weight of SQUIM's PESQ estimate")
    a = ap.parse_args()
    takes_dir, out = Path(a.takes_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(a.list)
    takes = {r["key"]: sorted(takes_dir.glob(f"{r['key']}_t*.wav")) for r in rows}
    from prep_voice import ecapa, embed

    spk = ecapa()
    hold = set((TARGET / "tts" / "holdout_ids.txt").read_text(encoding="utf-8").splitlines())
    train = [r for r in read_csv(TARGET / "train.csv") if r["id"] not in hold]
    cen = np.mean([embed(spk, clip_path(r["id"])) for r in random.Random(1).sample(train, min(300, len(train)))], 0)
    cen /= np.linalg.norm(cen)
    asr = whisper_model()
    mean_bank = np.load(TARGET / "emo_bank.npz")["feats"].mean(0)
    ab = read_csv(TARGET / "arousal_bank.csv")
    num = lambda v: float(v) if v not in ("", None) else np.nan  # noqa: E731
    sd = np.nanstd(np.array([[num(r[k]) for k in AROUSAL] for r in ab]), 0) + 1e-9
    w = np.array([0.0 if k == "level" else v for k, v in AROUSAL.items()])
    ebank = ending_bank(train) if a.w_ending else None
    squim = None
    if a.w_quality:
        import os

        import torch
        import torchaudio

        os.environ.setdefault("TORCH_HOME", str(Path(r"E:/voice-lab-cache/torch")))
        squim = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().eval()
    work = out / "_work"
    work.mkdir(exist_ok=True)
    emo = emotions([p for r in rows for p in takes[r["key"]]] + [Path(r["ref_path"]) for r in rows], work)
    report = []
    for r in rows:
        ts = takes[r["key"]]
        if not ts:
            continue
        fr, lr = emo[str(Path(r["ref_path"]))]
        cr = fr - mean_bank
        ar = arousal_raw(Path(r["ref_path"]), r["ref_text"])
        e_cos, same, adist, timbre, cers, ends, pesq = [], [], [], [], [], [], []
        for t in ts:
            ft, lt = emo[str(t)]
            ct = ft - mean_bank
            e_cos.append(float(ct @ cr / (np.linalg.norm(ct) * np.linalg.norm(cr) + 1e-9)))
            same.append(float(lt == lr))
            at = arousal_raw(t, r["text"])
            if at and ar:
                d = np.array([num(at[k]) - num(ar[k]) for k in AROUSAL]) / sd
                ok = np.isfinite(d) & (w > 0)
                adist.append(float(np.sqrt(np.sum(w[ok] * d[ok] ** 2) / max(w[ok].sum(), 1e-9))))
            else:
                adist.append(9.0)
            timbre.append(float(embed(spk, t) @ cen))
            cers.append(cer(r["text"], transcribe(asr, t)[0]))
            ends.append(ending_score(final_shape(t), r["text"], ebank) if ebank else 0.0)
            if squim is not None:
                import prosody
                from vl_common import read_mono

                x, sr = read_mono(t)
                with torch.inference_mode():
                    pesq.append(float(squim(torch.from_numpy(np.ascontiguousarray(prosody._resample(x, sr), dtype=np.float32))[None])[1]))
            else:
                pesq.append(0.0)
        score = (z(e_cos) + a.w_cat * np.array(same) - z(adist) + a.w_timbre * z(timbre)
                 - 30 * np.maximum(0, np.array(cers) - 0.1) + a.w_ending * z(ends) + a.w_quality * z(pesq))
        k = int(np.argmax(score))
        shutil.copy2(ts[k], out / f"{r['key']}.wav")
        report.append({"key": r["key"], "pick": ts[k].name, "score": round(float(score[k]), 2), "ref_cat": lr,
                       "cats": " ".join(emo[str(t)][1] for t in ts),
                       "emotion_cos": " ".join(f"{x:.2f}" for x in e_cos), "same_cat": " ".join(f"{x:.0f}" for x in same),
                       "arousal_dist": " ".join(f"{x:.2f}" for x in adist), "timbre": " ".join(f"{x:.3f}" for x in timbre),
                       "cer": " ".join(f"{x:.2f}" for x in cers), "ending": " ".join(f"{x:.2f}" for x in ends),
                       "pesq": " ".join(f"{x:.2f}" for x in pesq)})
        print(f"{r['key']} -> {ts[k].name}", flush=True)
    write_csv(out / "pick.csv", report)
    shutil.rmtree(work, ignore_errors=True)
    print(f"picked {len(report)} lines -> {out}")
