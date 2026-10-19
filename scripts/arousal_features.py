"""Arousal cues of speech clips, for the emotion matching (tools env: parselmouth).

emotion2vec names the emotion's category but barely its intensity: the clips it matched to the game phrases were
hardly closer in pitch, loudness or pace than random ones (2026-10-15). Per clip:
  pitch    median pitch (semitones re 100 Hz)          range   10-90 % pitch spread (semitones)
  level    90th percentile of 10 ms frame levels (dB)  rate    kana/kanji per voiced second (rough pace)
  tilt     voiced-frame energy 1-4 kHz over 0-1 kHz (dB): vocal effort, tense vs soft
Raw values; emotion_match.py compares them z-scored within each corpus (her clips / the game phrases), so a
studio-vs-anime level or channel difference does not count.

  python scripts/arousal_features.py <list.csv> <out.csv>
list.csv: id,path,text rows. Rows of out.csv whose wav is unchanged (size + mtime) are reused.
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np

import prosody
from vl_common import read_mono

FIELDS = ["id", "stamp", "pitch", "range", "level", "rate", "tilt"]


def features(path, text):
    wav, sr = read_mono(path)
    st, xs = prosody.semitones(wav, sr)
    voiced = np.isfinite(st)
    if voiced.sum() < 5:
        return None
    v = st[voiced]
    h = int(0.01 * sr)
    frames = wav[: len(wav) // h * h].reshape(-1, h)
    lv = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-9)
    # voiced frames for the tilt: pitch frame times -> 10 ms blocks of the waveform
    idx = np.clip(np.round(xs[voiced] / 0.01).astype(int), 0, len(frames) - 1)
    spec = np.abs(np.fft.rfft(frames[idx] * np.hanning(h), axis=1)) ** 2
    f = np.fft.rfftfreq(h, 1 / sr)
    low, mid = spec[:, (f >= 80) & (f < 1000)].sum(), spec[:, (f >= 1000) & (f < 4000)].sum()
    chars = len(re.sub(r"[^぀-ヿ一-鿿]", "", text or ""))
    return {"pitch": float(np.median(v)), "range": float(np.percentile(v, 90) - np.percentile(v, 10)),
            "level": float(np.percentile(lv, 90)), "rate": chars / max(voiced.sum() * 0.01, 0.2) if chars else "",
            "tilt": float(10 * np.log10((mid + 1e-12) / (low + 1e-12)))}


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    with open(src, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    old = {}
    if out.exists():
        with open(out, newline="", encoding="utf-8-sig") as fh:
            old = {r["id"]: r for r in csv.DictReader(fh)}
    res, done = [], 0
    for r in rows:
        p = Path(r["path"])
        stamp = f"{p.stat().st_size}-{p.stat().st_mtime_ns}"
        if r["id"] in old and old[r["id"]]["stamp"] == stamp:
            res.append(old[r["id"]])
            continue
        ft = features(p, r.get("text", ""))
        res.append({"id": r["id"], "stamp": stamp, **(ft or {k: "" for k in FIELDS[2:]})})
        done += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(res)
    print(f"arousal features: {done} computed, {len(res) - done} reused -> {out}")
