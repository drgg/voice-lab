"""Diagnose the non-verbal lines (grunts, breaths, shouts) of the current set (tools env).

Per line, the final output against the original performance:
  dur_ratio     length vs the original
  level_db      RMS level difference (loudness matching cannot measure clips under 0.4 s)
  voiced_o/f    share of loud frames with a pitch: a breath that comes back voiced sounds buzzy
  st_shift      pitch shift in semitones (RVC was asked for +2)
  hf_o/f        share of energy above 8 kHz in loud frames: noise-like hiss / metallic sheen
  sr_change     how much AudioSR changed the RVC output (spectral distance 40 kHz vs 48 kHz)
Writes eval/<set>/nonverbal_check.csv with a 问题 column per line and group (make_choice_page.py shows them).
Usage:  python scripts/check_nonverbal.py [--groups final_48k,N_rvc_48k,...]
"""
import argparse

import numpy as np
import parselmouth
import soundfile as sf

from vl_common import EVAL, EXP, load_lines, read_mono, rms_db, write_csv
from vl_paths import routes


def voiced_share(wav, sr):
    snd = parselmouth.Sound(wav.astype(np.float64), sr)
    f0 = snd.to_pitch_ac(time_step=0.005, pitch_floor=75, pitch_ceiling=900).selected_array["frequency"]
    inten = snd.to_intensity(time_step=0.005, minimum_pitch=75).values[0]
    n = min(len(f0), len(inten))
    if n == 0:
        return float("nan"), float("nan")
    f0, inten = f0[:n], inten[:n]
    loud = inten > inten.max() - 25
    v = f0[loud] > 0
    med = float(np.median(f0[loud][v])) if v.sum() >= 3 else float("nan")
    return float(v.mean()) if loud.any() else float("nan"), med


def spectrum(wav, sr, n_fft=1024):
    hop = n_fft // 4
    if len(wav) < n_fft:
        wav = np.pad(wav, (0, n_fft - len(wav)))
    frames = np.lib.stride_tricks.sliding_window_view(wav, n_fft)[::hop] * np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    e = spec.sum(axis=1)
    loud = e >= np.percentile(e, 50)
    return spec[loud].mean(axis=0), np.fft.rfftfreq(n_fft, 1 / sr)


def hf_share(wav, sr):
    s, f = spectrum(wav, sr)
    return float(s[f >= 8000].sum() / (s.sum() + 1e-12))


def spectral_distance(a, sr_a, b, sr_b):
    """Mean |dB| difference of the average spectra up to 20 kHz (both resampled onto 100 Hz bins)."""
    sa, fa = spectrum(a, sr_a)
    sb, fb = spectrum(b, sr_b)
    grid = np.arange(100, 20000, 100)
    da = 10 * np.log10(np.interp(grid, fa, sa) + 1e-12)
    db = 10 * np.log10(np.interp(grid, fb, sb) + 1e-12)
    return float(np.mean(np.abs((da - da.max()) - (db - db.max()))))


def check(ln, path, pre_sr=None):
    """Measures and Chinese issue notes for one output of one line (pre_sr: the same output before AudioSR)."""
    o, so = read_mono(ln["wav"])
    f, sf_ = read_mono(path)
    vo, mo = voiced_share(o, so)
    vf, mf = voiced_share(f, sf_)
    row = {"id": ln["id"], "src_wav": ln.get("src_wav", ""), "text": ln["ref_text"],
           "dur": round(len(o) / so, 2), "dur_ratio": round((len(f) / sf_) / (len(o) / so), 2),
           "level_db": round(rms_db(f) - rms_db(o), 1),
           "voiced_o": round(vo, 2), "voiced_f": round(vf, 2),
           "st_shift": round(12 * np.log2(mf / mo), 1) if np.isfinite(mo) and np.isfinite(mf) else "",
           "hf_o": round(hf_share(o, so), 3), "hf_f": round(hf_share(f, sf_), 3),
           "sr_change": round(spectral_distance(*read_mono(pre_sr), f, sf_), 1)
           if pre_sr is not None and pre_sr.exists() else ""}
    issues = []
    if abs(row["level_db"]) > 4:
        issues.append(f"音量差{row['level_db']:+.0f}dB")
    if np.isfinite(vo) and np.isfinite(vf) and vf - vo > 0.25:
        issues.append(f"气声变成浊音({vo:.0%}→{vf:.0%})")
    if np.isfinite(vo) and np.isfinite(vf) and vo - vf > 0.25:
        issues.append(f"浊音丢失({vo:.0%}→{vf:.0%})")
    if row["st_shift"] != "" and abs(row["st_shift"]) > 4.5:  # an octave slip of the pitch tracker
        issues.append(f"音高跳变{row['st_shift']:+.1f}半音")
    if row["hf_f"] > 2 * row["hf_o"] + 0.02:
        issues.append(f"高频噪声({row['hf_o']:.2f}→{row['hf_f']:.2f})")
    if not 0.9 <= row["dur_ratio"] <= 1.1:
        issues.append(f"时长比{row['dur_ratio']:.2f}")
    # only meaningful for full-band sources (RVC, 40 kHz); 22 kHz TTS output is meant to be extended
    if row["sr_change"] != "" and row["sr_change"] > 6 and sf.info(str(pre_sr)).samplerate >= 32000:
        issues.append(f"超分改动大({row['sr_change']:.0f}dB)")
    row["问题"] = "；".join(issues)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="final_48k", help="comma list of exp/ groups to check")
    a = ap.parse_args()

    route = routes()
    rows = []
    for g in a.groups.split(","):
        pre = EXP / g[:-4] if g.endswith("_48k") else None  # the same group before it was taken to 48 kHz
        mine = []
        for ln in load_lines():
            out = EXP / g / f"{ln['id']}.wav"
            if route.get(ln["id"]) != "nonverbal" or not out.exists():
                continue
            mine.append({"group": g, **check(ln, out, pre / f"{ln['id']}.wav" if pre else None)})
        counts = {}
        for r in mine:
            for k in filter(None, r["问题"].split("；")):
                key = k.split("(")[0].rstrip("0123456789.+-dB半音")
                counts[key] = counts.get(key, 0) + 1
        print(f"{g}: {len(mine)} lines, {sum(bool(r['问题']) for r in mine)} flagged {counts}")
        rows += mine
    write_csv(EVAL / "nonverbal_check.csv", rows)
    print(f"-> {EVAL / 'nonverbal_check.csv'}")
