"""Objective metrics for every experiment group (tools env).

Per output line:
  dev_span_pct   speech-span length vs the original line (what matters for putting it back in game)
  cer            Whisper (ja) transcript of the output vs the line text, both normalised to hiragana
  f0_med/f0_iqr  pitch register and range (Praat)
  f0_toward_target   0 = stayed at the source character's register, 1 = moved all the way to the target voice's register
  lufs           integrated loudness
  cos_target/cos_source/spk_pref  ECAPA similarity - reference only: both are the same actress
  f0_corr        intonation fidelity: pitch contour vs the original, DTW-aligned (prosody.py)
  range_ratio    pitch range (semitones) vs the original - below ~0.7 sounds flat
Writes eval/metrics.csv, eval/metrics_summary.csv and eval/metrics_by_bucket.csv (short / medium /
long lines separately). Usage:  python scripts/metrics.py
"""
import numpy as np
import parselmouth
import pyloudnorm

import prosody
from prep_voice import ecapa, embed
from vl_common import EVAL, EXP, SOURCE_DIR, LINES, METHODS, TARGET, load_lines, normalize_ja, read_mono, \
    speech_span, transcribe, whisper_model, write_csv
from vl_paths import char_label


def f0_stats(wav, sr):
    pitch = parselmouth.Sound(wav.astype(np.float64), sr).to_pitch_ac(time_step=0.01, pitch_floor=75, pitch_ceiling=900)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    if len(f0) < 5:
        return float("nan"), float("nan")
    return float(np.median(f0)), float(np.percentile(f0, 75) - np.percentile(f0, 25))


def lufs(wav, sr):
    if len(wav) < int(0.5 * sr):
        return float("nan")
    return float(pyloudnorm.Meter(sr).integrated_loudness(wav))


def cer(ref, hyp):
    import jiwer

    r, h = normalize_ja(ref), normalize_ja(hyp)
    return float(jiwer.cer(r, h)) if r else float("nan")


def centroid(model, paths):
    embs = [embed(model, p) for p in paths]
    c = np.mean(embs, axis=0)
    return c / np.linalg.norm(c)


def main():
    lines = load_lines()
    refs = TARGET / "refs"
    zt_clips = sorted(refs.glob("abx_*.wav")) + [refs / "ref_12s.wav"]
    spk = ecapa()
    c_target = np.load(TARGET / "anchor_centroid.npy") if (TARGET / "anchor_centroid.npy").exists() \
        else centroid(spk, zt_clips)
    source_pool = sorted((SOURCE_DIR / "wav").glob("*.wav"))[:120] or [l["wav"] for l in lines]
    c_source = centroid(spk, source_pool)
    f0_target = np.nanmedian([f0_stats(*read_mono(p))[0] for p in zt_clips])
    asr = whisper_model()

    rows = []
    groups = [("orig", LINES)] + [(m, EXP / m) for m in METHODS if any((EXP / m).glob("L*.wav"))]
    for method, folder in groups:
        for ln in lines:
            out = folder / f"{ln['id']}.wav"
            if not out.exists():
                continue
            o_wav, o_sr = read_mono(ln["wav"])
            wav, sr = read_mono(out)
            span_o, span = speech_span(o_wav, o_sr), speech_span(wav, sr)
            text, _, _ = transcribe(asr, out)
            f0_o = f0_stats(o_wav, o_sr)[0]
            f0_med, f0_iqr = f0_stats(wav, sr)
            e = embed(spk, out)
            denom = f0_target - f0_o
            rows.append({
                "method": method, "id": ln["id"], "bucket": ln.get("bucket", ""),
                "dur_orig": round(len(o_wav) / o_sr, 3), "dur_out": round(len(wav) / sr, 3),
                "span_orig": round(span_o, 3), "span_out": round(span, 3),
                "dev_span_pct": round(100 * (span - span_o) / span_o, 1) if span_o else "",
                "cer": round(cer(ln["ref_text"], text), 3),
                "f0_med": round(f0_med, 1), "f0_iqr": round(f0_iqr, 1),
                "f0_toward_target": round((f0_med - f0_o) / denom, 2) if abs(denom) > 5 else "",
                "f0_corr": round(prosody.contour_corr(o_wav, o_sr, wav, sr), 3),
                "range_ratio": round(prosody.range_ratio(o_wav, o_sr, wav, sr), 2),
                "lufs": round(lufs(wav, sr), 1),
                "cos_target": round(float(e @ c_target), 3), "cos_source": round(float(e @ c_source), 3),
                "spk_pref": round(float(e @ c_target - e @ c_source), 3),
                "asr_text": text, "ref_text": ln["ref_text"],
            })
            print(f"{method} {ln['id']} cer={rows[-1]['cer']} dev={rows[-1]['dev_span_pct']}%")
    write_csv(EVAL / "metrics.csv", rows)

    def summarize(rs, method):
        dev = np.array([abs(r["dev_span_pct"]) for r in rs if r["dev_span_pct"] != ""], dtype=float)
        tow = np.array([r["f0_toward_target"] for r in rs if r["f0_toward_target"] != ""], dtype=float)
        return {
            "method": method, "label": METHODS.get(method, f"{char_label()}原声"), "n": len(rs),
            "mean_abs_dev_pct": round(float(dev.mean()), 1) if len(dev) else "",
            "within_5pct": round(float((dev <= 5).mean()), 2) if len(dev) else "",
            "mean_cer": round(float(np.nanmean([r["cer"] for r in rs])), 3),
            "median_f0": round(float(np.nanmedian([r["f0_med"] for r in rs])), 1),
            "median_f0_toward_target": round(float(np.median(tow)), 2) if len(tow) else "",
            "mean_f0_corr": round(float(np.nanmean([r["f0_corr"] for r in rs])), 3),
            "mean_range_ratio": round(float(np.nanmean([r["range_ratio"] for r in rs])), 2),
            "mean_spk_pref": round(float(np.mean([r["spk_pref"] for r in rs])), 3),
        }

    fields = ["method", "label", "n", "mean_abs_dev_pct", "within_5pct", "mean_cer", "median_f0",
              "median_f0_toward_target", "mean_f0_corr", "mean_range_ratio", "mean_spk_pref"]
    summary, by_bucket = [], []
    for method, _ in groups:
        rs = [r for r in rows if r["method"] == method]
        if not rs:
            continue
        summary.append(summarize(rs, method))
        for b in ("short", "fill", "medium", "long"):
            if [r for r in rs if r["bucket"] == b]:
                by_bucket.append({"bucket": b, **summarize([r for r in rs if r["bucket"] == b], method)})
    summary.append({"method": "ref_target", "label": "目标声线参考", "median_f0": round(float(f0_target), 1)})
    write_csv(EVAL / "metrics_summary.csv", summary, fieldnames=fields)
    write_csv(EVAL / "metrics_by_bucket.csv", by_bucket, fieldnames=["bucket"] + fields)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
