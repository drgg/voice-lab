"""Off-pitch check of the finished lines of the current voice (tools env).

Per line of final_48k (final.csv says where it came from):
  jumps     pitch leaps of more than 5 semitones between neighbouring voiced frames (10 ms) - squeaks,
            octave errors; a voice glides, it does not jump
  octave    frames more than 8 semitones off the running median of their 90 ms neighbourhood
  rvc_off   for RVC versions: frames where the output's pitch left its input's (plus the intended shift) by
            more than 2 semitones, and the longest such run - RVC's pitch tracker losing a breathy or quiet
            stretch is the usual cause of a note sung off
Lines over the limits are flagged, worst first, to eval/<set>_<voice>/pitch_check.csv.

  envs/tools/Scripts/python scripts/check_pitch.py          (VL_SET, VL_VOICE)
"""
import numpy as np
from scipy.ndimage import median_filter

import prosody
from vl_common import EVAL, EXP, LINES, read_csv, read_mono, voice_cfg, write_csv

PRE = {"D4_indextts_phrase_rvc_48k": ("D4_indextts_phrase", 0), "N_rvc_48k": ("orig", 0),
       "N_rvc2_48k": ("orig", None), "V_rvc_48k": ("orig", None)}


def track(path):
    wav, sr = read_mono(path)
    st, xs = prosody.semitones(wav, sr)
    return st, xs, len(wav) / sr


def jumps(st):
    d = np.abs(np.diff(st))
    ok = np.isfinite(d)
    return int(np.sum(d[ok] > 5.0))


def octave(st):
    v = np.isfinite(st)
    if v.sum() < 9:
        return 0
    s = np.where(v, st, np.nanmedian(st))
    med = median_filter(s, size=9, mode="nearest")
    return int(np.sum(v & (np.abs(s - med) > 8.0)))


def rvc_off(st, xs, pre_path, shift):
    sp, xp, _ = track(pre_path)
    sp_at = np.interp(xs, xp, np.nan_to_num(sp, nan=-999.0), left=-999.0, right=-999.0)
    both = np.isfinite(st) & (sp_at > -900)
    if both.sum() < 10:
        return 0.0, 0
    bad = both & (np.abs(st - sp_at - shift) > 2.0)
    run = longest = 0
    for b in bad:
        run = run + 1 if b else 0
        longest = max(longest, run)
    return float(bad.sum() / both.sum()), longest


if __name__ == "__main__":
    nv_shift = float(voice_cfg().get("nonverbal_pitch", 0))
    rows = []
    for r in read_csv(EXP / "final_48k" / "final.csv"):
        f = EXP / "final_48k" / f"{r['id']}.wav"
        st, xs, dur = track(f)
        row = {"id": r["id"], "route": r["route"], "source": r["source"], "seconds": round(dur, 2),
               "voiced": int(np.isfinite(st).sum()), "jumps": jumps(st), "octave": octave(st),
               "rvc_off": "", "rvc_off_run_ms": "", "text": r["text"][:30]}
        pre = PRE.get(r["source"])
        if pre:
            src, shift = pre
            p = LINES / f"{r['id']}.wav" if src == "orig" else EXP / src / f"{r['id']}.wav"
            if p.exists():
                frac, run = rvc_off(st, xs, p, nv_shift if shift is None else shift)
                row["rvc_off"], row["rvc_off_run_ms"] = round(frac, 3), run * 10
        row["flag"] = bool(row["jumps"] >= 2 or row["octave"] >= 5
                           or (row["rvc_off"] != "" and row["rvc_off"] > 0.08 and row["rvc_off_run_ms"] >= 50))
        rows.append(row)
    sev = lambda r: (r["flag"], (r["rvc_off"] or 0) * 10 + r["jumps"] + r["octave"] / 5)  # noqa: E731
    rows.sort(key=sev, reverse=True)
    out = EVAL / "pitch_check.csv"
    write_csv(out, rows)
    flagged = [r for r in rows if r["flag"]]
    print(f"{len(flagged)} of {len(rows)} lines flagged -> {out}")
    for r in flagged:
        print(f"  {r['id']} {r['source']:28s} jumps {r['jumps']:2d} octave {r['octave']:3d} "
              f"rvc_off {r['rvc_off']!s:6s} run {r['rvc_off_run_ms']!s:4s} ms  {r['text']}")
