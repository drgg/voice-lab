"""Turn episode videos (--src <folder>, read in place) into candidate clips of the target voice (VL_VOICE).

Stages (run all, or one with --stage); <voice> = data/<VL_VOICE> (default default):
  extract   ffmpeg: pick the Japanese audio stream -> <voice>/audio/<ep>.wav (44.1k stereo)
  separate  audio-separator (Roformer): vocals + instrumental -> <voice>/vocals/
  slice     Silero VAD on vocals -> <voice>/slices/<ep>_<startms>.wav + per-clip metrics
  transcribe faster-whisper (ja) text per clip, to help spotting her lines
  page      eval/pick_anchors.html - listen and mark a few clips that are surely her
  rank      ECAPA cosine of every clip to the anchors -> <voice>/slices_ranked.csv
  classify  speaker classifier from speech-habit cues -> p_target in slices_ranked.csv
Defaults for the cues (catchphrase, own / other characters' markers, pitch floor, the episodes where the
catchphrase is only hers) come from <voice>/voice.json (vl_paths.voice_cfg).

Usage (tools env):  set VL_VOICE=<voice> & python scripts/prep_voice.py [--stage all|extract|...] --src <folder>
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from vl_common import (CACHE, EVAL, OTHER_MARKERS, TARGET, read_csv, read_mono, rms_db, transcribe, whisper_model,
                       write_csv)
from vl_paths import page_path, voice_cfg

CFG = voice_cfg()

SRC, AUDIO, VOCALS, SLICES = (TARGET / d for d in ("src", "audio", "vocals", "slices"))
VIDEO_EXT = {".mkv", ".mp4", ".m2ts", ".ts", ".webm", ".avi", ".mov", ".flac", ".wav"}
SEP_MODEL = "vocals_mel_band_roformer.ckpt"  # Kim MelBand Roformer, best vocal SDR in audio-separator's list
SLICE_SR = 44100


def stage_extract(src):
    AUDIO.mkdir(parents=True, exist_ok=True)
    skip = CFG.get("exclude_sources")  # e.g. recap specials: scenes already in the episodes
    videos = sorted(p for p in Path(src).iterdir() if p.suffix.lower() in VIDEO_EXT
                    and not (skip and re.search(skip, p.name)))
    if not videos:
        raise SystemExit(f"no episode files in {src}")
    for v in videos:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=index:stream_tags=language,title", "-of", "json", str(v)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        streams = json.loads(probe.stdout or "{}").get("streams", [])
        pick = 0
        for k, s in enumerate(streams):
            tags = {key.lower(): str(val).lower() for key, val in (s.get("tags") or {}).items()}
            if tags.get("language") in ("jpn", "ja") or "日本" in tags.get("title", ""):
                pick = k
                break
        out = AUDIO / f"{v.stem}.wav"
        if out.exists() or (VOCALS / f"{v.stem}_vocals.wav").exists():
            continue  # incremental: episodes arrive in batches (the audio is dropped once separated)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(v), "-map", f"0:a:{pick}", "-ac", "2",
             "-ar", str(SLICE_SR), "-c:a", "pcm_s16le", str(out)],
            check=True,
        )
        print(f"{v.name}: audio stream {pick} of {len(streams)} -> {out.name}")


def stage_separate(model):
    from audio_separator.separator import Separator

    VOCALS.mkdir(parents=True, exist_ok=True)
    sep = Separator(output_dir=str(VOCALS), model_file_dir=str(CACHE / "separator"), output_format="WAV")
    sep.load_model(model_filename=model)
    for a in sorted(AUDIO.glob("*.wav")):
        if (VOCALS / f"{a.stem}_vocals.wav").exists():
            continue
        outputs = sep.separate(str(a))
        for o in outputs:
            o = VOCALS / Path(o).name
            stem = "vocals" if "(vocals)" in o.name.lower() else "inst"
            o.replace(VOCALS / f"{a.stem}_{stem}.wav")
        if (VOCALS / f"{a.stem}_vocals.wav").exists() and (VOCALS / f"{a.stem}_inst.wav").exists():
            a.unlink()  # only separation needs it (~0.25 GB an episode; extract can redo it from the video)
        print(f"separated {a.name}")


def hf_ratio_db(x, sr, cutoff=4000):
    """Energy above `cutoff` relative to total; band-limited radio/phone voices sit very low."""
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    return 10 * np.log10(spec[freqs >= cutoff].sum() / (spec.sum() + 1e-12) + 1e-12)


def load_vad():
    """torch.jit.load(path) goes through C++ fopen, which fails under the non-ASCII project path;
    loading from bytes avoids it."""
    import io

    import silero_vad
    import torch

    jit = Path(silero_vad.__file__).parent / "data" / "silero_vad.jit"
    model = torch.jit.load(io.BytesIO(jit.read_bytes()), map_location="cpu")
    model.eval()
    return model


def release_offset(ref_vocals, vocals, at=300.0, length=60.0):
    """Seconds to add to `ref_vocals` times to land on the same moment in `vocals` (another release of the
    episode): envelope cross-correlation over +-10 s, then sample-exact. 0 when the reference is missing."""
    from scipy.signal import correlate

    if not Path(ref_vocals).exists():
        return 0.0
    sr = sf.info(str(vocals)).samplerate
    a = sf.read(str(ref_vocals), start=int(at * sr), stop=int((at + length) * sr), dtype="float32", always_2d=True)[0].mean(1)
    b = sf.read(str(vocals), start=int((at - 10) * sr), stop=int((at + length + 10) * sr), dtype="float32",
                always_2d=True)[0].mean(1)
    h = sr // 1000
    env = lambda x: np.abs(x[: len(x) // h * h]).reshape(-1, h).mean(1)  # noqa: E731
    ea, eb = env(a), env(b)
    k = int(np.argmax(correlate(eb - eb.mean(), ea - ea.mean(), mode="valid"))) * h
    lo = max(0, k - 2 * h)
    k = lo + int(np.argmax(correlate(b[lo:k + 2 * h + sr * 10], a[:sr * 10], mode="valid")))
    return k / sr - 10.0


def stage_slice(clips_from=None, add_cuts_from=None):
    """Speech clips of every separated episode (Silero VAD). With `clips_from` (another voice folder holding the
    same episodes from another release, e.g. a lossy one): its clip times and transcripts are reused, matched by
    episode number, so only the audio changes (no VAD, no Whisper; the releases must share the timing).
    With `add_cuts_from`: VAD clips as usual, plus that folder's clips re-cut from these vocals (ids ending in
    "x", transcripts carried over) - two segmentations of the same speech; build_trainset keeps one per span."""
    clips_from = clips_from or add_cuts_from
    import torch
    from silero_vad import get_speech_timestamps

    vad = load_vad()
    SLICES.mkdir(parents=True, exist_ok=True)
    # carry over per-clip results (transcripts, pitch) from earlier runs; slicing is deterministic
    prev = {r["id"]: r for r in read_csv(TARGET / "slices.csv")} if (TARGET / "slices.csv").exists() else {}
    by_ep = {}
    if clips_from:
        for r in read_csv(TARGET.parent / clips_from / "slices.csv"):
            by_ep.setdefault(episode_no(r["id"]), []).append(r)
    rows = []
    for voc in sorted(VOCALS.glob("*_vocals.wav")):
        ep = voc.stem[: -len("_vocals")]
        wav, sr = read_mono(voc)
        inst, _ = read_mono(VOCALS / f"{ep}_inst.wav") if (VOCALS / f"{ep}_inst.wav").exists() else (None, sr)
        stamps = []
        if not clips_from or add_cuts_from:
            wav16, _ = read_mono(voc, sr=16000)
            # voice.json may cut finer: a shorter pause splits quick exchanges into one speaker per clip
            stamps = get_speech_timestamps(
                torch.from_numpy(wav16), vad, sampling_rate=16000, threshold=0.5,
                min_speech_duration_ms=800, max_speech_duration_s=CFG.get("vad_max_speech_s", 12),
                min_silence_duration_ms=CFG.get("vad_min_silence_ms", 300), speech_pad_ms=120, return_seconds=True,
            )
        if clips_from:
            src = by_ep.get(episode_no(ep), [])
            if not src and not add_cuts_from:
                print(f"{ep}: no episode {episode_no(ep)} in {clips_from} - skipped")
                continue
            if src:
                off = release_offset(TARGET.parent / clips_from / "vocals" / f"{src[0]['episode']}_vocals.wav", voc)
                print(f"{ep}: {len(src)} clips from {clips_from}, timing offset {off * 1000:+.0f} ms")
                stamps += [{"start": max(0.0, float(r["start"]) + off), "end": float(r["end"]) + off, "src": r}
                           for r in src]
        for st in stamps:
            a, b = int(st["start"] * sr), int(st["end"] * sr)
            seg = wav[a:b]
            cid = f"{ep}_{int(st['start'] * 1000):08d}" + ("x" if add_cuts_from and "src" in st else "")
            if "src" in st:  # same words, same pitch: the other release's transcript stands
                prev.setdefault(cid, {"text": st["src"].get("text", ""), "f0_med": st["src"].get("f0_med", "")})
            sf.write(SLICES / f"{cid}.wav", seg, sr, subtype="PCM_16")
            voc_db = rms_db(seg)
            inst_db = rms_db(inst[a:b]) if inst is not None else -120.0
            rows.append({
                "id": cid, "episode": ep, "start": round(st["start"], 2), "end": round(st["end"], 2),
                "duration": round((b - a) / sr, 2), "rms_db": round(voc_db, 1),
                "bg_ratio_db": round(inst_db - voc_db, 1), "hf_ratio_db": round(hf_ratio_db(seg, sr), 1),
            })
        print(f"{ep}: {len(stamps)} clips")
    for r in rows:
        r["radio"] = float(r["hf_ratio_db"]) < -38
        old = prev.get(r["id"], {})
        r["text"], r["f0_med"] = old.get("text", ""), old.get("f0_med", "")
    write_csv(TARGET / "slices.csv", rows)


def stage_transcribe():
    rows = read_csv(TARGET / "slices.csv")
    todo = [r for r in rows if not r.get("text")]
    if not todo:
        return
    model = whisper_model()
    for k, r in enumerate(todo, 1):
        r["text"], _, _ = transcribe(model, SLICES / f"{r['id']}.wav")
        if k % 500 == 0:
            print(f"  transcribed {k}/{len(todo)}", flush=True)
    write_csv(TARGET / "slices.csv", rows)
    print(f"transcribed {len(todo)} new clips")


def candidates(rows):
    return [
        r for r in rows
        if 1.5 <= float(r["duration"]) <= 10 and float(r["bg_ratio_db"]) < -6 and r["radio"] != "True"
    ]


def stage_page():
    rows = candidates(read_csv(TARGET / "slices.csv"))
    items = [
        {"id": r["id"], "ep": r["episode"], "t": float(r["start"]), "d": float(r["duration"]),
         "text": r.get("text", ""), "src": page_path(TARGET / "slices" / f"{r['id']}.wav")}
        for r in rows
    ]
    tpl = (Path(__file__).parent / "templates" / "pick_anchors.html").read_text(encoding="utf-8")
    EVAL.mkdir(exist_ok=True)
    (EVAL / "pick_anchors.html").write_text(
        tpl.replace("__ITEMS__", json.dumps(items, ensure_ascii=False)).replace("__VOICE__", CFG["label"]),
        encoding="utf-8"
    )
    print(f"{len(items)} candidate clips -> {EVAL / 'pick_anchors.html'}")


def episode_no(cid):
    """Episode number in a clip id ("..._-_05_...", "... - 05 - ...", "... Title 05 [..." or "... 05 (BDRip..."),
    else None. A season-and-episode name ("... S02E01-...") counts on from the season's start, given in the voice's
    settings: `season_offsets` {"S01": 0, "S02": 12} (a release that numbers straight through has no seasons)."""
    m = re.search(r"S(\d{2})E(\d{2,3})", cid)
    if m:
        off = (CFG.get("season_offsets") or {}).get(f"S{m.group(1)}")
        return int(m.group(2)) + off if off is not None else None
    m = re.search(r"(?:_-_| - )(\d{2,3})(?:_|\s|\(|$)", cid) or re.search(r"\s(\d{2,3})\s*[\[(]", cid)
    return int(m.group(1)) if m else None


def in_anchor_eps(cid):
    """True if the voice's settings allow cue-based positives from this clip: `anchor_match` (a regex on the
    clip id, e.g. one season's episodes) or `anchor_episodes` ([first, last] episode number)."""
    if CFG.get("anchor_match"):
        return re.search(CFG["anchor_match"], cid) is not None
    eps = CFG.get("anchor_episodes")
    if not eps:
        return True
    n = episode_no(cid)
    return n is not None and eps[0] <= n <= eps[1]


def ecapa():
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    # COPY: Windows refuses the default symlink strategy without admin rights
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(CACHE / "ecapa"),
        run_opts={"device": "cuda"}, local_strategy=LocalStrategy.COPY,
    )


def embed(model, path):
    import torch

    wav, _ = read_mono(path, sr=16000)
    with torch.no_grad():
        e = model.encode_batch(torch.from_numpy(wav)[None].cuda()).squeeze().cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9)


def clip_f0(cid):
    import parselmouth

    wav, sr = read_mono(SLICES / f"{cid}.wav")
    pitch = parselmouth.Sound(wav.astype(np.float64), sr).to_pitch_ac(time_step=0.01, pitch_floor=75, pitch_ceiling=900)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    return round(float(np.median(f0)), 1) if len(f0) > 5 else ""


def stage_rank(anchors_path, anchor_dir=None, keyword=None, min_f0=0.0, bootstrap=15):
    """Rank clips by speaker similarity to 02.

    Anchor sources, in order of reliability:
      marks     clips marked in pick_anchors.html
      keyword   clips whose transcript contains a catchphrase only she says (target: "<catchphrase>"),
                limited to short clips (<=3 s) to avoid multi-speaker segments
      anchor_dir clean recordings of the same actress. NOTE: with game voice vs. anime mix this
                picked the male lead - ECAPA does not bridge that channel gap; prefer the others.
    `min_f0` drops clips below that median pitch (e.g. 180 Hz removes male characters) before
    ranking. The top `bootstrap` clips then become anchors for a second pass.
    """
    model = ecapa()
    rows = read_csv(TARGET / "slices.csv")
    for k, r in enumerate(rows, 1):
        if r.get("f0_med", "") in ("", None):
            r["f0_med"] = clip_f0(r["id"])
        if k % 1000 == 0:
            print(f"  pitch {k}/{len(rows)}", flush=True)
    voiced = lambda r: r["f0_med"] not in ("", "nan") and float(r["f0_med"]) >= min_f0  # noqa: E731
    embs = {}
    for k, r in enumerate(rows, 1):
        embs[r["id"]] = embed(model, SLICES / f"{r['id']}.wav")
        if k % 1000 == 0:
            print(f"  embedded {k}/{len(rows)}", flush=True)
    pos, neg = [], []
    if keyword:
        pos = [r["id"] for r in rows if re.search(keyword, r.get("text") or "") and voiced(r)
               and float(r["duration"]) <= 3.0 and r["radio"] != "True" and in_anchor_eps(r["id"])]
        seed = np.mean([embs[k] for k in pos], axis=0)
        seed /= np.linalg.norm(seed)
        usable = [r for r in rows if voiced(r) and float(r["duration"]) >= 1.2 and r["radio"] != "True"]
        first = sorted(usable, key=lambda r: -float(embs[r["id"]] @ seed))
        pos = list(dict.fromkeys(pos + [r["id"] for r in first[:bootstrap]]))
        print(f"keyword '{keyword}': seed + bootstrap anchors = {len(pos)}")
    elif anchor_dir:
        refs = sorted(Path(anchor_dir).glob("*.wav"))
        ref_embs = [embed(model, p) for p in refs if sf.info(str(p)).duration >= 1.5]
        centroid = np.mean(ref_embs, axis=0)
        centroid /= np.linalg.norm(centroid)
        usable = [r for r in rows if float(r["duration"]) >= 1.5 and r["radio"] != "True"]
        first = sorted(usable, key=lambda r: -float(embs[r["id"]] @ centroid))
        pos = [r["id"] for r in first[:bootstrap]]
        print(f"anchor-dir: {len(ref_embs)} reference files; bootstrap anchors: {pos}")
    else:
        marks = json.loads(Path(anchors_path).read_text(encoding="utf-8"))
        pos = [k for k, v in marks.items() if v in (1, True, "1", "yes")]
        neg = [k for k, v in marks.items() if v in (0, False, "0", "no")]
    if len(pos) < 3:
        raise SystemExit("need at least 3 anchors: mark clips in pick_anchors.html, or pass --keyword / --anchor-dir")
    centroid = np.mean([embs[k] for k in pos], axis=0)
    centroid /= np.linalg.norm(centroid)
    neg_c = np.mean([embs[k] for k in neg], axis=0) if neg else None
    for r in rows:
        # clips under the pitch floor keep their similarity but are pushed out of the reference pool
        r["score"] = round(float(embs[r["id"]] @ centroid) - (0 if voiced(r) else 1.0), 4)
        r["neg_score"] = round(float(embs[r["id"]] @ neg_c), 4) if neg_c is not None else ""
        r["anchor"] = 1 if r["id"] in pos else (0 if r["id"] in neg else "")
    rows.sort(key=lambda r: -r["score"])
    write_csv(TARGET / "slices_ranked.csv", rows)
    np.save(TARGET / "anchor_centroid.npy", centroid)
    print(f"ranked {len(rows)} clips; anchors +{len(pos)} / -{len(neg)}")


def stage_classify(own_markers, min_f0, other_markers=OTHER_MARKERS):
    """Speaker classifier on ECAPA embeddings, trained from transcript cues (run after `rank`).

    Cosine-to-centroid cannot separate the target from the other girls well (same age range and register).
    Weak labels from speech habits fix that: female clips whose transcript has the character's own
    markers (e.g. her first-person pronoun / a catchphrase) are positives, clips with other characters' markers (私, わよ, polite
    です/ます, ...) are negatives. A logistic regression on the embeddings then scores every clip;
    the probability is written as `p_target` into slices_ranked.csv.
    """
    import re

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    if not own_markers or not other_markers:  # an empty pattern matches every transcript: no labels at all
        raise SystemExit(f"set own_markers and other_markers in {TARGET / 'voice.json'} (her / other characters' speech habits)")
    rows = read_csv(TARGET / "slices_ranked.csv")
    model = ecapa()
    female = [r for r in rows if r["f0_med"] not in ("", "nan") and float(r["f0_med"]) >= min_f0
              and float(r["duration"]) >= 1.0 and r["radio"] != "True"]
    embs = {}
    for k, r in enumerate(female, 1):
        embs[r["id"]] = embed(model, SLICES / f"{r['id']}.wav")
        if k % 1000 == 0:
            print(f"  embedded {k}/{len(female)}", flush=True)
    own, other = re.compile(own_markers), re.compile(other_markers)
    pos = [r["id"] for r in female if own.search(r.get("text") or "") and not other.search(r.get("text") or "")
           and in_anchor_eps(r["id"])]
    neg = [r["id"] for r in female if other.search(r.get("text") or "") and not own.search(r.get("text") or "")]
    X = np.stack([embs[i] for i in pos + neg])
    y = np.array([1] * len(pos) + [0] * len(neg))
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    acc = cross_val_score(clf, X, y, cv=5, scoring="balanced_accuracy")
    clf.fit(X, y)
    print(f"weak labels: {len(pos)} positive / {len(neg)} negative; 5-fold balanced accuracy {acc.mean():.3f}")
    for r in rows:
        r["p_target"] = round(float(clf.predict_proba(embs[r["id"]][None])[0, 1]), 4) if r["id"] in embs else 0.0
    write_csv(TARGET / "slices_ranked.csv", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "extract", "separate", "slice", "transcribe", "page", "rank", "classify"])
    ap.add_argument("--own-markers", default=CFG["own_markers"], help="regex of the target character's speech habits")
    ap.add_argument("--src", default=str(SRC), help="folder with the episode files (read in place, not copied)")
    ap.add_argument("--model", default=SEP_MODEL)
    ap.add_argument("--anchors", default=str(EVAL / "anchors.json"))
    ap.add_argument("--anchor-dir", default=None, help="clean recordings of the same actress to use as anchors")
    ap.add_argument("--keyword", default=CFG["keyword"],
                    help="catchphrase only the target character says ('' = use the marks of pick_anchors.html)")
    ap.add_argument("--min-f0", type=float, default=CFG["min_f0"], help="drop clips below this median pitch (Hz)")
    ap.add_argument("--clips-from", default=CFG.get("clips_from"),
                    help="slice: reuse the clip times + transcripts of this voice folder (same episodes, other release)")
    a = ap.parse_args()
    if a.stage in ("all", "extract"):
        stage_extract(a.src)
    if a.stage in ("all", "separate"):
        stage_separate(a.model)
    if a.stage in ("all", "slice"):
        stage_slice(a.clips_from, CFG.get("add_cuts_from"))
    if a.stage in ("all", "transcribe"):
        stage_transcribe()
    if a.stage in ("all", "page"):
        stage_page()
    if a.stage == "rank":
        stage_rank(a.anchors, a.anchor_dir, a.keyword or None, a.min_f0)
    if a.stage == "classify":
        stage_classify(a.own_markers, a.min_f0)
