"""Refine one line, step 1: cut it into phrases with hand-corrected text (tools env).

For lines the automatic pass gets wrong - e.g. L14 is not 「…」 but a wounded
「…」 [0.8 s] 「…」 (breathy) [0.4 s] 「…」, which Whisper had collapsed into one
sentence. The line is cut at its pauses (>= --min-gap s), one piece per '|'-separated text, into
<set>/refine/<line>/ in the same phrases.csv format that run_indextts.py --phrases reads.
scan_lines.py suggests the texts (routes.csv, column `suggest`).

Usage:  envs/tools/Scripts/python scripts/refine_split.py --line L14 --texts "<phrase 1>|<phrase 2>|<phrase 3>"
"""
import argparse

import soundfile as sf

from segment_lines import silences, speech_bounds
from vl_common import LINES, REFINE, read_mono, write_csv

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", required=True)
    ap.add_argument("--texts", required=True, help="phrase texts separated by |")
    ap.add_argument("--min-gap", type=float, default=0.2)
    a = ap.parse_args()

    texts = [t.strip() for t in a.texts.split("|")]
    wav, sr = read_mono(LINES / f"{a.line}.wav")
    dur = len(wav) / sr
    gaps = sorted(silences(wav, sr, min_gap=a.min_gap), key=lambda g: g[0] - g[1])[:len(texts) - 1]
    if len(gaps) != len(texts) - 1:
        raise SystemExit(f"{a.line}: found {len(gaps)} pauses >= {a.min_gap}s, need {len(texts) - 1}")
    cuts = [0.0] + sorted((g[0] + g[1]) / 2 for g in gaps) + [dur]  # the longest pauses, in time order

    out = REFINE / a.line
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, text in enumerate(texts):
        c0, c1 = cuts[k], cuts[k + 1]
        seg = wav[int(c0 * sr):int(c1 * sr)]
        s, e = speech_bounds(seg, sr, c0)
        pid = f"{a.line}_{k + 1}"
        sf.write(out / f"{pid}.wav", seg, sr, subtype="PCM_16")
        rows.append({"id": pid, "line_id": a.line, "idx": k + 1, "text": text, "cut_start": round(c0, 3),
                     "cut_end": round(c1, 3), "speech_start": round(s, 3), "speech_end": round(e, 3)})
        print(f"{pid}: {text}  speech {s:.2f}-{e:.2f}s")
    write_csv(out / "phrases.csv", rows)
