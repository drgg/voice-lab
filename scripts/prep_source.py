"""Prepare a game character's lines from the exported voice files in data/<char>/wem/ (VL_CHAR; default source, 原角色).

1. probe every Wwise file with vgmstream and keep codec / sample rate / channels
   (needed later to re-encode converted lines with matching settings) -> wem_info.csv
2. decode to wav, one file per source (bank subsongs become separate files) -> data/<char>/wav/
3. transcribe with faster-whisper large-v3 (ja) -> transcript.csv
4. pick N lines spread over short / medium / long durations -> data/<char>/lines/{L01..}.wav + lines.csv

Usage (tools env):  [set VL_CHAR=other &] python scripts/prep_source.py [--n 20] [--skip-decode]
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from vl_common import SOURCE_DIR, normalize_ja, read_csv, read_mono, rms_db, transcribe, whisper_model, write_csv

WEM_DIR = SOURCE_DIR / "wem"
WAV_DIR = SOURCE_DIR / "wav"
LINES_DIR = SOURCE_DIR / "lines"
AUDIO_EXT = {".wem", ".bnk", ".ogg", ".opus", ".wav", ".flac"}
# (bucket, min_s, max_s, share of N)
BUCKETS = [("short", 0.8, 2.5, 0.3), ("medium", 2.5, 6.0, 0.4), ("long", 6.0, 14.0, 0.3)]


def vgm_info(path, subsong=None):
    cmd = ["vgmstream-cli", "-m", "-I"]
    if subsong is not None:
        cmd += ["-s", str(subsong)]
    out = subprocess.run(cmd + [str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = out.stdout.strip()
    start = text.find("{")
    if start < 0:
        return {}
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        return {}


def decode_all():
    files = sorted(p for p in WEM_DIR.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    if not files:
        raise SystemExit(f"no audio files under {WEM_DIR} - export the source character's voice files there first")
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in files:
        rel = src.relative_to(WEM_DIR)
        base = rel.with_suffix("").as_posix().replace("/", "__")
        info = vgm_info(src)
        subsongs = int((info.get("streamInfo") or {}).get("total") or 1)
        for idx in range(1, subsongs + 1):
            sub = idx if subsongs > 1 else None
            meta = vgm_info(src, sub) if sub else info
            name = f"{base}__s{idx:03d}" if sub else base
            out = WAV_DIR / f"{name}.wav"
            cmd = ["vgmstream-cli", "-o", str(out)] + (["-s", str(sub)] if sub else []) + [str(src)]
            res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            ok = res.returncode == 0 and out.exists()
            stream = meta.get("streamInfo") or {}
            rows.append({
                "wav": out.name if ok else "",
                "source": rel.as_posix(),
                "subsong": sub or "",
                "stream_name": stream.get("name", ""),
                "encoding": meta.get("encoding", ""),
                "layout": meta.get("layout", ""),
                "sample_rate": meta.get("sampleRate", ""),
                "channels": meta.get("channels", ""),
                "num_samples": meta.get("numberOfSamples", ""),
                "bitrate": meta.get("bitrate", ""),
                "decode_ok": ok,
            })
        print(f"decoded {rel} ({subsongs} stream(s))")
    write_csv(SOURCE_DIR / "wem_info.csv", rows)
    print(f"wem_info.csv: {len(rows)} streams, {sum(r['decode_ok'] for r in rows)} decoded")


def transcribe_all():
    model = whisper_model()
    rows = []
    for w in sorted(WAV_DIR.glob("*.wav")):
        wav, sr = read_mono(w)
        text, no_speech, logprob = transcribe(model, w)
        rows.append({
            "wav": w.name,
            "duration": round(len(wav) / sr, 3),
            "rms_db": round(rms_db(wav), 1),
            "text": text,
            "no_speech_prob": round(no_speech, 3),
            "avg_logprob": round(logprob, 3),
        })
        print(f"{w.name}\t{rows[-1]['duration']}s\t{text}")
    write_csv(SOURCE_DIR / "transcript.csv", rows)


def pick_lines(n):
    from rapidfuzz import fuzz

    rows = read_csv(SOURCE_DIR / "transcript.csv")
    # the line text drives the TTS group and the CER metric, so demand a confident transcript;
    # long lines carry more words, so they get a stricter bar
    usable = [
        r for r in rows
        if r["text"] and float(r["no_speech_prob"]) < 0.5
        and float(r["avg_logprob"]) > (-0.3 if float(r["duration"]) >= 6.0 else -0.55)
    ]
    chosen = []

    def is_dup(r):
        # the same line often ships twice (e.g. two takes / variants) and Whisper spells them
        # differently, so compare hiragana readings with a fuzzy ratio instead of exact text
        key = normalize_ja(r["text"])
        return any(fuzz.ratio(key, normalize_ja(c["text"])) >= 70 for c in chosen)

    for bucket, lo, hi, share in BUCKETS:
        pool = [r for r in usable if lo <= float(r["duration"]) < hi]
        pool.sort(key=lambda r: float(r["rms_db"]))  # quiet -> loud, spread across intensity
        k = min(len(pool), round(n * share))
        # evenly spaced picks first; if one is a duplicate, fall back to the rest of the bucket
        order = [int(i) for i in np.linspace(0, len(pool) - 1, k).round()] if k else []
        order += [i for i in range(len(pool)) if i not in order]
        taken = 0
        for i in order:
            if taken >= k:
                break
            if not is_dup(pool[i]):
                chosen.append({**pool[i], "bucket": bucket})
                taken += 1
    # top up from whatever is left if a bucket was short
    for r in usable:
        if len(chosen) >= n:
            break
        if 0.8 <= float(r["duration"]) < 14.0 and not is_dup(r):
            chosen.append({**r, "bucket": "fill"})

    if LINES_DIR.exists():
        shutil.rmtree(LINES_DIR)
    LINES_DIR.mkdir(parents=True)
    out = []
    for i, r in enumerate(chosen[:n], 1):
        lid = f"L{i:02d}"
        shutil.copy2(WAV_DIR / r["wav"], LINES_DIR / f"{lid}.wav")
        out.append({
            "id": lid, "src_wav": r["wav"], "bucket": r["bucket"],
            "duration": r["duration"], "text": r["text"], "text_edit": "",
        })
    write_csv(LINES_DIR / "lines.csv", out)
    print(f"picked {len(out)} lines -> {LINES_DIR / 'lines.csv'} (fill text_edit to correct transcripts)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--skip-transcribe", action="store_true")
    a = ap.parse_args()
    if not a.skip_decode:
        decode_all()
    if not a.skip_transcribe:
        transcribe_all()
    pick_lines(a.n)
