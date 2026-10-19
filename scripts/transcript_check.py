"""Second opinion on the transcripts of the target voice's clips (tools env; VL_VOICE).

The clips' texts come from Whisper large-v3; where it misheard, a text-to-speech model trained on them learns the
wrong words and a re-dub reads them (the voice's test line T18: 「…」 written 「…」, the
generated take sounded wrong - the user heard it, 2026-10-15). kotoba-whisper v2.0 (Kotoba Technologies' Japanese
Whisper, faster-whisper format, cache/whisper/kotoba-whisper-v2.0-faster) transcribes every clip again; the two
texts are compared after normalize_ja (punctuation and spacing ignored). A large disagreement marks a clip to leave
out or check by ear. Writes data/<voice>/transcripts_check.csv.

  set VL_VOICE=myvoice & python scripts/transcript_check.py [--limit 0.25]
"""
import argparse

from vl_common import CACHE, TARGET, ensure_cuda_dlls, normalize_ja, read_csv, write_csv
from emotion_match import clip_path


def cer(a, b):
    import jiwer

    a, b = normalize_ja(a), normalize_ja(b)
    return float(jiwer.cer(a, b)) if a else (0.0 if not b else 1.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=float, default=0.25, help="CER between the two above which a clip is flagged")
    a = ap.parse_args()
    ensure_cuda_dlls()
    from faster_whisper import WhisperModel

    model = WhisperModel(str(CACHE / "whisper" / "kotoba-whisper-v2.0-faster"), device="cuda", compute_type="float16")
    rows = []
    train = read_csv(TARGET / "train.csv")
    for k, r in enumerate(train, 1):
        segs, _ = model.transcribe(str(clip_path(r["id"])), language="ja", beam_size=5, condition_on_previous_text=False,
                                   chunk_length=15)
        alt = "".join(s.text for s in segs).strip()
        d = cer(r["text"], alt)
        rows.append({"id": r["id"], "file": r["file"], "whisper": r["text"], "kotoba": alt, "cer": round(d, 3),
                     "flag": "disagree" if d > a.limit else ""})
        if k % 100 == 0:
            print(f"  {k}/{len(train)}", flush=True)
    write_csv(TARGET / "transcripts_check.csv", rows)
    n = sum(1 for r in rows if r["flag"])
    print(f"{n} of {len(rows)} clips disagree above CER {a.limit} -> {TARGET / 'transcripts_check.csv'}")
