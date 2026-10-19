"""Non-verbal lines, step 1: the "keep the original" option and the IndexTTS inputs (tools env).

Grunts and breaths carry little of the character (same actress), and RVC on a 0.2 s grunt often loses
its voicing or adds hiss, so the original performance itself is one of the options. Lines whose
text is a usable interjection (「…」「…」「…」「…」, up to --max-tts seconds) also get an IndexTTS
take, as one phrase per line (on L14 the listener picked a TTS 「…」 over voice conversion);
hallucinated or empty texts do not.

Writes exp/<set>/N_orig_48k/<id>.wav and <set>/nonverbal_phrases/ (phrases.csv + wavs).
Usage:  envs/tools/Scripts/python scripts/prep_nonverbal.py
"""
import argparse
import shutil

from scan_lines import HALLUCINATION, INTERJECTION
from segment_lines import speech_bounds
from vl_common import EXP, load_lines, normalize_ja, read_mono, write_csv
from vl_paths import SET_DIR, routes

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tts", type=float, default=4.0)
    a = ap.parse_args()

    route = routes()
    orig, phr = EXP / "N_orig_48k", SET_DIR / "nonverbal_phrases"
    orig.mkdir(parents=True, exist_ok=True)
    phr.mkdir(parents=True, exist_ok=True)
    rows, n = [], 0
    for ln in load_lines():
        if route.get(ln["id"]) != "nonverbal":
            continue
        n += 1
        shutil.copy2(ln["wav"], orig / f"{ln['id']}.wav")  # exported game audio is already 48 kHz
        wav, sr = read_mono(ln["wav"])
        kana = normalize_ja(ln["ref_text"])
        if not kana or not INTERJECTION.fullmatch(kana) or HALLUCINATION.search(ln["ref_text"]) \
                or len(wav) / sr > a.max_tts:
            continue
        s, e = speech_bounds(wav, sr, 0.0)
        shutil.copy2(ln["wav"], phr / f"{ln['id']}_1.wav")
        rows.append({"id": f"{ln['id']}_1", "line_id": ln["id"], "idx": 1, "text": ln["ref_text"],
                     "cut_start": 0.0, "cut_end": round(len(wav) / sr, 3),
                     "speech_start": round(s, 3), "speech_end": round(e, 3)})
    write_csv(phr / "phrases.csv", rows)
    print(f"{n} non-verbal lines -> {orig}; {len(rows)} with an interjection text for IndexTTS -> {phr}")
