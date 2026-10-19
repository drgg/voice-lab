"""Phrase-level pitch check of the re-dubbed lines (tools env): the usual cause of a line that sounds off-key.

The re-dub picks each phrase's take on its own and places it on the original's timing, so one phrase can come
out noticeably higher or lower than its neighbours, or with a melody of its own. Per phrase (phrases.csv) of
every final line from the re-dub:
  step     how far the phrase's pitch sits from the rest of the line, minus the same in the original
           (semitones; the original may legitimately rise or drop, the output should do the same)
  corr     melody (contour) correlation with the original phrase
Phrases with |step| >= 2.5 st, or corr < 0.2 on a phrase with 0.6 s or more of voice, are flagged; the lines
are written worst first to eval/<set>_<voice>/phrase_pitch.csv, the phrases to phrase_pitch_phrases.csv.

  envs/tools/Scripts/python scripts/check_phrase_pitch.py     (VL_SET, VL_VOICE)
"""
import numpy as np

import prosody
from vl_common import EVAL, EXP, LINES, PHRASES, read_csv, read_mono, write_csv

STEP_ST, MIN_CORR = 2.5, 0.2


def seg(wav, sr, a, b):
    return wav[int(a * sr):int(b * sr)]


if __name__ == "__main__":
    final = {r["id"]: r for r in read_csv(EXP / "final_48k" / "final.csv")}
    by_line = {}
    for p in read_csv(PHRASES / "phrases.csv"):
        by_line.setdefault(p["line_id"], []).append(p)
    prow, lrow = [], []
    for lid, phs in by_line.items():
        f = final.get(lid)
        if not f or not f["source"].startswith("D4_") or len(phs) < 2:
            continue
        o, osr = read_mono(LINES / f"{lid}.wav")
        y, ysr = read_mono(EXP / "final_48k" / f"{lid}.wav")
        med_o, med_y, voiced, corr = [], [], [], []
        for p in phs:
            a, b = float(p["speech_start"]), float(p["speech_end"])
            so, sy = seg(o, osr, a, b), seg(y, ysr, a, b)
            st_o, _ = prosody.semitones(so, osr)
            st_y, _ = prosody.semitones(sy, ysr)
            med_o.append(np.nanmedian(st_o) if np.isfinite(st_o).sum() >= 5 else np.nan)
            med_y.append(np.nanmedian(st_y) if np.isfinite(st_y).sum() >= 5 else np.nan)
            voiced.append(np.isfinite(st_y).sum() * prosody.STEP)
            corr.append(prosody.contour_corr(so, osr, sy, ysr) if len(so) > osr * 0.3 and len(sy) > ysr * 0.3 else np.nan)
        mo, my = np.array(med_o), np.array(med_y)
        ok = np.isfinite(mo) & np.isfinite(my)
        if ok.sum() < 2:
            continue
        rel_o, rel_y = mo - np.nanmedian(mo[ok]), my - np.nanmedian(my[ok])
        worst = 0.0
        bad = []
        for k, p in enumerate(phs):
            step = rel_y[k] - rel_o[k] if ok[k] else np.nan
            flag = bool((np.isfinite(step) and abs(step) >= STEP_ST)
                        or (np.isfinite(corr[k]) and corr[k] < MIN_CORR and voiced[k] >= 0.6))
            prow.append({"line": lid, "phrase": p["id"], "idx": p["idx"], "text": p["text"],
                         "step_st": round(float(step), 2) if np.isfinite(step) else "",
                         "corr": round(float(corr[k]), 2) if np.isfinite(corr[k]) else "",
                         "voiced_s": round(float(voiced[k]), 2), "flag": flag})
            if flag:
                bad.append(p["idx"])
                worst = max(worst, abs(step) if np.isfinite(step) else 0, (1 - corr[k]) * 2 if np.isfinite(corr[k]) else 0)
        lrow.append({"line": lid, "phrases": len(phs), "flagged": ",".join(bad), "worst": round(worst, 2),
                     "text": f["text"][:40]})
    lrow.sort(key=lambda r: (-(len(r["flagged"]) > 0), -r["worst"]))
    write_csv(EVAL / "phrase_pitch.csv", lrow)
    write_csv(EVAL / "phrase_pitch_phrases.csv", prow)
    flagged = [r for r in lrow if r["flagged"]]
    print(f"{len(flagged)} of {len(lrow)} multi-phrase re-dubbed lines have off phrases -> {EVAL / 'phrase_pitch.csv'}")
    for r in flagged:
        print(f"  {r['line']} phrases {r['flagged']:8s} of {r['phrases']:2d}  worst {r['worst']:5.2f}  {r['text']}")
