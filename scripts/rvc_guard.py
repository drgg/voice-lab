"""RVC pitch guard for the re-dubbed lines (tools env).

RVC keeps the input's melody, except where its pitch tracker (RMVPE) loses a breathy or very short stretch:
there the phrase comes out an octave off (a short murmur dropped 16.7 st once). Per phrase of every RVC'd line,
the median pitch of the RVC output is compared with its input's (same span):
  check   lines with a phrase moved by more than 3 st -> printed as a comma list (for run_rvc.py --lines)
  pick    for those lines, the --alt group (the same lines through another pitch tracker, FCPE) replaces the
          RVC output where it moved less; the replaced file is newer, so the 48 kHz steps redo just it.
          Phrases still off with both trackers are printed as "FALLBACK <phrase ids>": the take itself is the
          problem (a murmur the re-dub cannot say)
  splice  for those phrases, the original performance converted by RVC (run_rvc.py on the phrase wavs, the
          voice's register shift) is pasted over the phrase's span, at the span's loudness, 10 ms crossfades

  envs/tools/Scripts/python scripts/rvc_guard.py check [--src D4_indextts_phrase --rvc D4_indextts_phrase_rvc]
  envs/tools/Scripts/python scripts/rvc_guard.py pick --lines L154,...
  envs/tools/Scripts/python scripts/rvc_guard.py splice --phrases L154_8 --vc D4_phrase_vc
"""
import argparse
import json
import shutil

import librosa
import numpy as np
import soundfile as sf

import prosody
from vl_common import EXP, PHRASES, read_csv, read_mono

LIMIT = 3.0


def moved(src, out, phrases):
    """(largest |pitch move| in st over the line's phrases, ids of the phrases moved past LIMIT)."""
    a, sa = read_mono(src)
    b, sb = read_mono(out)
    worst, bad = 0.0, []
    for p in phrases:
        s, e = float(p["speech_start"]), float(p["speech_end"])
        ma = prosody.median_st(a[int(s * sa):int(e * sa)], sa)
        mb = prosody.median_st(b[int(s * sb):int(e * sb)], sb)
        if np.isfinite(ma) and np.isfinite(mb):
            worst = max(worst, abs(mb - ma))
            if abs(mb - ma) > LIMIT:
                bad.append(p["id"])
    return worst, bad


def splice(line_path, seg, sr_seg, start, end, fade=0.01):
    """Paste `seg` over [start, end) s of the line at the span's loudness, with crossfades."""
    y, sr = sf.read(str(line_path), dtype="float32")
    seg = librosa.resample(seg, orig_sr=sr_seg, target_sr=sr) if sr_seg != sr else seg
    seg, _ = librosa.effects.trim(seg, top_db=40)
    a, b = int(start * sr), min(len(y), int(end * sr))
    old = y[a:b]
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))) + 1e-9)  # noqa: E731
    seg = seg * (rms(old) / rms(seg))
    n = int(fade * sr)
    out = y.copy()
    hole = np.ones(b - a, np.float32)
    hole[:n], hole[len(hole) - n:] = np.linspace(1, 0, n), np.linspace(0, 1, n)
    out[a:b] *= 1 - hole if len(hole) > 2 * n else 1          # the bad take fades out over the span
    k = min(len(seg), len(out) - a)
    s2 = seg[:k].copy()
    s2[:n] *= np.linspace(0, 1, n)
    s2[k - n:] *= np.linspace(1, 0, n)
    out[a:a + k] += s2
    sf.write(str(line_path), np.clip(out, -0.99, 0.99), sr, subtype="PCM_16")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "pick", "splice"])
    ap.add_argument("--src", default="D4_indextts_phrase")
    ap.add_argument("--rvc", default="D4_indextts_phrase_rvc")
    ap.add_argument("--alt", default="D4_indextts_phrase_rvc_alt")
    ap.add_argument("--vc", default="D4_phrase_vc")
    ap.add_argument("--lines", default="")
    ap.add_argument("--phrases", default="")
    a = ap.parse_args()
    by_line, by_id = {}, {}
    for p in read_csv(PHRASES / "phrases.csv"):
        by_line.setdefault(p["line_id"], []).append(p)
        by_id[p["id"]] = p
    src, rvc, alt = EXP / a.src, EXP / a.rvc, EXP / a.alt
    if a.cmd == "check":
        bad = {}
        for f in sorted(rvc.glob("L*.wav")):
            if (src / f.name).exists() and f.stem in by_line:
                m, _ = moved(src / f.name, f, by_line[f.stem])
                if m > LIMIT:
                    bad[f.stem] = round(m, 2)
        (rvc / "guard.json").write_text(json.dumps(bad, indent=1), encoding="utf-8")
        print(",".join(bad) if bad else "")
    elif a.cmd == "pick":
        fallback = []
        for lid in filter(None, a.lines.split(",")):
            m_rvc, bad_rvc = moved(src / f"{lid}.wav", rvc / f"{lid}.wav", by_line[lid])
            m_alt, bad_alt = moved(src / f"{lid}.wav", alt / f"{lid}.wav", by_line[lid]) if (alt / f"{lid}.wav").exists() \
                else (99, bad_rvc)
            use = m_alt < m_rvc
            if use:
                shutil.copy2(alt / f"{lid}.wav", rvc / f"{lid}.wav")
                (rvc / f"{lid}.wav").touch()
            still = bad_alt if use else bad_rvc
            fallback += still
            print(f"{lid}: rmvpe moved {m_rvc:.1f} st, other tracker {m_alt:.1f} st -> {'other' if use else 'rmvpe'}"
                  + (f"; still off: {','.join(still)}" if still else ""))
        print("FALLBACK " + ",".join(fallback) if fallback else "FALLBACK")
    else:
        for pid in filter(None, a.phrases.split(",")):
            p = by_id[pid]
            vc = EXP / a.vc / f"{pid}.wav"
            seg, sr = read_mono(vc)
            splice(rvc / f"{p['line_id']}.wav", seg, sr, float(p["speech_start"]), float(p["speech_end"]))
            print(f"{p['line_id']}: phrase {pid} 「{p['text']}」 <- the original performance through RVC")
