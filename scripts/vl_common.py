"""Shared helpers for the voice-lab scripts (tools env)."""
import csv
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

from vl_paths import CHAR, CHAR_DIR, EVAL, EXP, LINES, PHRASES, REFINE, ROOT, VOICE, VOICE_DIR, voice_cfg  # noqa: F401  (set- and voice-aware)

DATA = ROOT / "data"
SOURCE_DIR = CHAR_DIR  # the game character's exported lines (wem/, wav/, transcript.csv, wem_info.csv), whatever the
                   # set - named after the first character (VL_CHAR picks another)
TARGET = VOICE_DIR  # the target voice's material (vl_paths: VL_VOICE)
CACHE = ROOT / "cache"


def shift_text(st):
    """Semitones as the pages say them (env.ps1 Format-Shift says it the same way)."""
    st = float(st)
    return f"升 {st:g} 半音" if st > 0 else f"降 {-st:g} 半音" if st < 0 else "不变调"


_NV = shift_text(voice_cfg().get("nonverbal_pitch", 2))  # the voice's register shift for the original performance

METHODS = {
    "A_seedvc_timbre": "Seed-VC V2 只换音色",
    "B_seedvc_style": "Seed-VC V2 + 风格转换",
    "C_vevo2": "Vevo2 风格转换 VC",
    "D_indextts25": "IndexTTS-2.5 重新配音",
    # round 2: per-line emotion-matched the target references (scripts/emotion_match.py)
    "A2_seedvc_emo": "Seed-VC V2 只换音色 · 情绪匹配参考",
    "D2_indextts_emo": "IndexTTS-2.5 · 情绪匹配参考 · 情感强度 1.0",
    # round 2 + AudioSR super-resolution 22.05 kHz -> 48 kHz (scripts/superres.py)
    "A2_seedvc_emo_48k": "Seed-VC V2 只换音色 · 情绪匹配 · 48kHz 超分",
    "D2_indextts_emo_48k": "IndexTTS-2.5 · 情绪匹配 · 48kHz 超分",
    # round 3: whole series -> larger cleaned emotion bank + Seed-VC V2 fine-tuned on the target
    "A3_seedvc_ft": "Seed-VC V2 微调 · 只换音色 · 情绪匹配",
    "B3_seedvc_ft_style": "Seed-VC V2 微调 · 音色+风格转换 · 情绪匹配",
    "D3_indextts_emo": "IndexTTS-2.5 · 全集情绪库",
    "A3_seedvc_ft_48k": "Seed-VC V2 微调 · 只换音色 · 48kHz",
    "B3_seedvc_ft_style_48k": "Seed-VC V2 微调 · 音色+风格转换 · 48kHz",
    "D3_indextts_emo_48k": "IndexTTS-2.5 · 全集情绪库 · 48kHz",
    # round 3b: RVC (Applio) trained on the target set (scripts/run_rvc.py)
    "E_rvc": "RVC 直接转换",
    "D3_indextts_emo_rvc": "IndexTTS-2.5 + RVC 精修（两段式）",
    "E_rvc_48k": "RVC 直接转换 · 48kHz",
    "D3_indextts_emo_rvc_48k": "IndexTTS-2.5 + RVC 精修 · 48kHz",
    # round 4: per-phrase re-dub with best-of-4 takes (segment_lines / select_phrases) + RVC;
    # RVC on the original performance shifted to the target voice's register
    "D4_indextts_phrase": "IndexTTS-2.5 · 分句 · 4 选 1",
    "D4_indextts_phrase_rvc": "IndexTTS-2.5 · 分句 · 4 选 1 + RVC 精修",
    "E2_rvc_pitch2": "RVC 直接转换 · 升 2 半音",
    "D4_indextts_phrase_48k": "IndexTTS-2.5 · 分句 · 4 选 1 · 48kHz",
    "D4_indextts_phrase_rvc_48k": "IndexTTS-2.5 · 分句 · 4 选 1 + RVC 精修 · 48kHz",
    "E2_rvc_pitch2_48k": "RVC 直接转换 · 升 2 半音 · 48kHz",
    # production (run_d4.ps1): non-verbal lines by RVC on the original, hand-refined lines, final mix
    "V_rvc": f"原声表演 → RVC（{_NV}）",
    "V_rvc_48k": f"原声表演 → RVC（{_NV}）· 48kHz",
    "refined": "单句精修",
    "refined_48k": "单句精修 · 48kHz",
    "final_48k": "最终版 · 48kHz",
    # non-verbal options (run_nonverbal.ps1): padded RVC without AudioSR, TTS of the interjection, original
    "N_rvc_48k": "拟声词 · RVC 新版（不变调）",
    "N_rvc2_48k": f"拟声词 · RVC 新版（{_NV}）",
    "N_tts_48k": "拟声词 · IndexTTS 重配",
    "N_orig_48k": "拟声词 · 原声保留",
}


def ensure_cuda_dlls():
    """ctranslate2 (faster-whisper) needs cuBLAS/cuDNN DLLs; reuse the ones shipped in torch/lib."""
    import torch

    lib = Path(torch.__file__).parent / "lib"
    if lib.is_dir():
        os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(lib))


def read_mono(path, sr=None):
    """Read audio as float32 mono; resample with librosa when sr is given."""
    wav, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr is not None and sr != file_sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
        file_sr = sr
    return wav, file_sr


def duration(path):
    info = sf.info(str(path))
    return info.frames / info.samplerate


def speech_span(wav, sr, top_db=40):
    """Duration of the non-silent span (leading/trailing silence trimmed)."""
    import librosa

    if len(wav) == 0:
        return 0.0
    _, idx = librosa.effects.trim(wav, top_db=top_db)
    return (idx[1] - idx[0]) / sr


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9)


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# other characters' speech habits (per voice, data/<voice>/voice.json); see prep_voice.stage_classify
OTHER_MARKERS = voice_cfg()["other_markers"]


# a second band as clean as the first (voice.json extra_band [min_p, min_score]): a confident classifier makes up
# for lower similarity, which short clips get (measured by speech-habit cues per band, see README)
EXTRA_BAND = voice_cfg().get("extra_band")


def is_target_clip(r, min_p=0.7, min_score=0.6, max_bg=-8.0, min_dur=1.0, max_dur=8.0, extra=True):
    """A ranked clip that is confidently the target character and clean enough to train on.

    Needs both the transcript-cue classifier (p_target) and centroid similarity (score): each one
    alone lets other girls' lines through; long clips are dropped because they often hold 2 speakers.
    """
    text = r.get("text") or ""
    p, s = float(r.get("p_target") or 0.0), float(r["score"])
    if not (r["radio"] != "True" and float(r["bg_ratio_db"]) <= max_bg and min_dur <= float(r["duration"]) <= max_dur
            and not (OTHER_MARKERS and re.search(OTHER_MARKERS, text))):  # an empty pattern would match every text
        return False
    return (p >= min_p and s >= min_score) or bool(extra and EXTRA_BAND and p >= EXTRA_BAND[0] and s >= EXTRA_BAND[1])


def load_lines(lines_dir=None):
    """The current set's lines (vl_paths), or those in lines_dir (vl_paths.char_paths); `text_edit` (user
    correction) wins over the Whisper text."""
    d = Path(lines_dir) if lines_dir else LINES
    rows = read_csv(d / "lines.csv")
    for r in rows:
        r["ref_text"] = (r.get("text_edit") or "").strip() or r["text"].strip()
        r["wav"] = str(d / f"{r['id']}.wav")
    return rows


# Whisper's stock phrases on near-silence (video outros); a 0.13 s grunt comes back as 「…」
HALLUCINATED = re.compile(r"ご視聴ありがとうございました|それではまた|チャンネル登録|お疲れ様でした")


def route_info():
    """id -> the scan_lines.py row (route, reasons, per-burst suggestion); {} before the first scan."""
    from vl_paths import ROUTES

    return {r["id"]: r for r in read_csv(ROUTES)} if ROUTES.exists() else {}


def display_text(ln, route_info=None):
    """The line's text for pages: transcripts scan_lines.py marked untrustworthy (too short, no real words) and
    Whisper's stock hallucinations are shown as what they are, a short sound without words."""
    text = ln.get("ref_text") or ln.get("text") or ""
    reasons = (route_info or {}).get("reasons", "")
    if "无可信文本" in reasons or HALLUCINATED.search(text) or not text.strip():
        return f"（短促发声 {float(ln.get('duration') or 0):.2f} 秒，无台词）"
    return text


_PUNCT = re.compile(r"[\s　、。，,．.！!？?…‥・「」『』（）()\[\]【】〜~ー―\-—♪☆★\"'“”‘’：:；;]+")


def normalize_ja(text):
    """NFKC, drop punctuation/space, and map kanji/katakana to hiragana reading for a fair CER."""
    text = _PUNCT.sub("", unicodedata.normalize("NFKC", text or ""))
    try:
        import pykakasi

        kks = normalize_ja._kks = getattr(normalize_ja, "_kks", None) or pykakasi.kakasi()
        text = "".join(item["hira"] for item in kks.convert(text))
    except ImportError:
        pass
    return text


def whisper_model():
    ensure_cuda_dlls()
    from faster_whisper import WhisperModel

    return WhisperModel(
        "large-v3", device="cuda", compute_type="float16", download_root=str(CACHE / "whisper")
    )


def transcribe(model, path):
    segments, info = model.transcribe(
        str(path), language="ja", beam_size=5, vad_filter=False, condition_on_previous_text=False
    )
    segs = list(segments)
    text = "".join(s.text for s in segs).strip()
    no_speech = max((s.no_speech_prob for s in segs), default=1.0)
    logprob = float(np.mean([s.avg_logprob for s in segs])) if segs else -9.0
    return text, no_speech, logprob
