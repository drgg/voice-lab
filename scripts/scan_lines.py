"""Route every line of the current set before the production run (tools env).

Most game voice files are not sentences: 109 of the source character's 180 lines are under 1.5 s, many are
0.1-0.2 s combat grunts that Whisper "hears" as 「…」, and some (L14) mix a
grunt, breathy words and long pauses. A text re-dub cannot voice those, so each line gets a route:
  nonverbal  grunt / breath / shout, or no trustworthy text  -> RVC on the original performance
  refine     speech, but with grunt bursts, mostly breathy voice, long inner pauses or a shaky
             transcript -> still re-dubbed, flagged for a listen and run_refine.ps1 if it fails
  speech     ordinary dialogue -> phrase-level re-dub (D4)
For refine lines the column `suggest` holds a per-burst transcript, ready for
run_refine.ps1 -Texts after checking it by ear.

Writes <set>/routes.csv (vl_paths.ROUTES). Usage:  envs/tools/Scripts/python scripts/scan_lines.py
"""
import re

import numpy as np
import parselmouth
import soundfile as sf

from segment_lines import silences
from vl_common import CACHE, ensure_cuda_dlls, load_lines, normalize_ja, read_mono, write_csv
from vl_paths import ROUTES

# Whisper's stock hallucinations on near-silence (YouTube outros), never real lines in this game
HALLUCINATION = re.compile(r"ご視聴|チャンネル登録|ありがとうございました|それではまた|おやすみなさい|字幕")
# only interjection sounds, any length: 「…」, but also 18 s of 「…」 or 「…」
INTERJECTION = re.compile(r"[あいうえおぁぃぅぇぉっんはひふへほくやゃゅょーぇ]+")


def voice_profile(wav, sr):
    """Share of loud frames without pitch (breathy / whispered) and the pitch of each loud stretch."""
    snd = parselmouth.Sound(wav.astype(np.float64), sr)
    f0 = snd.to_pitch_ac(time_step=0.01, pitch_floor=75, pitch_ceiling=900)
    inten = snd.to_intensity(time_step=0.01)
    t = np.arange(0.0, len(wav) / sr, 0.01)
    db = np.array([inten.get_value(x) for x in t], dtype=float)
    hz = np.array([f0.get_value_at_time(x) for x in t], dtype=float)
    loud = np.nan_to_num(db, nan=-300) > np.nanmax(db) - 25
    return float(np.mean(~np.isfinite(hz[loud]))) if loud.any() else 0.0, t, hz, loud


def bursts(wav, sr, min_gap=0.25):
    """Voiced stretches separated by >= min_gap s of silence (including exact digital zeros)."""
    gaps = silences(wav, sr, min_gap=min_gap)
    nz = np.flatnonzero(np.abs(wav) > 1e-4)
    if len(nz) == 0:
        return []
    edges, start = [], nz[0] / sr
    for g0, g1 in gaps:
        edges.append((start, g0))
        start = g1
    edges.append((start, nz[-1] / sr))
    return [(a, b) for a, b in edges if b - a > 0.05]


if __name__ == "__main__":
    ensure_cuda_dlls()
    from faster_whisper import WhisperModel

    asr = None
    rows, counts = [], {}
    for ln in load_lines():
        wav, sr = read_mono(ln["wav"])
        dur = len(wav) / sr
        text, kana = ln["ref_text"], normalize_ja(ln["ref_text"])
        lp = float(ln.get("avg_logprob") or 0.0)
        ns = float(ln.get("no_speech_prob") or 0.0)
        breathy, t, hz, loud = voice_profile(wav, sr)
        parts = bursts(wav, sr)
        grunts = 0
        for a, b in parts:  # short stretch that peaks high = effort sound (L14's 「…」 swings 200-480 Hz)
            sel = (t >= a) & (t <= b) & np.isfinite(hz)
            if b - a < 0.5 and sel.sum() >= 3 and np.percentile(hz[sel], 90) > 380:
                grunts += 1
        reasons = []
        if dur < 0.6:
            reasons.append("太短")
        if not kana or (dur < 2.0 and HALLUCINATION.search(text)):  # a 48 s monologue may say ありがとう…
            reasons.append("无可信文本")
        if kana and INTERJECTION.fullmatch(kana):
            reasons.append("仅感叹词")
        if reasons and not ln.get("text_edit"):
            route = "nonverbal"
        else:
            if grunts and len(parts) > 1 and dur < 8:  # in long monologues a high syllable is just emphasis
                reasons.append(f"含{grunts}段闷哼/喊声")
            if breathy > 0.4:
                reasons.append(f"气声占{breathy:.0%}")
            if len(parts) >= 3 and dur < 5:
                reasons.append(f"{len(parts)}段+长停顿")
            if lp < -0.45 and not ln.get("text_edit"):
                reasons.append(f"转写置信度低({lp:.2f})")
            route = "refine" if reasons else "speech"
        suggest = ""
        if route == "refine" and len(parts) > 1:
            if asr is None:
                asr = WhisperModel("large-v3", device="cuda", compute_type="float16",
                                   download_root=str(CACHE / "whisper"))
            pieces = []
            for a, b in parts:
                seg = np.concatenate([np.zeros(int(0.3 * sr)), wav[int(a * sr):int(b * sr)], np.zeros(int(0.3 * sr))])
                tmp = CACHE / "tmp" / "scan_burst.wav"
                tmp.parent.mkdir(parents=True, exist_ok=True)
                sf.write(tmp, seg.astype(np.float32), sr)
                segs, _ = asr.transcribe(str(tmp), language="ja", beam_size=5, condition_on_previous_text=False)
                pieces.append("".join(s.text for s in segs).strip() or "?")
            suggest = "|".join(pieces)
        counts[route] = counts.get(route, 0) + 1
        rows.append({"id": ln["id"], "src_wav": ln.get("src_wav", ""), "duration": round(dur, 2), "route": route,
                     "reasons": "；".join(reasons), "text": text, "suggest": suggest,
                     "bursts": len(parts), "breathy": round(breathy, 2), "logprob": lp, "no_speech": ns})
        print(f"{ln['id']} {dur:5.2f}s {route:9s} {'；'.join(reasons):30s} {text}  {suggest}")
    write_csv(ROUTES, rows)
    print(f"routes: {counts} -> {ROUTES}")
