"""Round 2: pick an emotion-matched target reference for every source line (run in envs/seedvc: has funasr).

Round 1 used one fixed target reference for all lines, so every output got the same emotional colour.
Here emotion2vec embeds each source line and every clean target clip; for each line the closest the target
clips (cosine in emotion space) are joined into a per-line reference:
  data/default/refs/emo/<id>.wav   ~8-12 s (Seed-VC target and IndexTTS speaker prompt)
  data/default/refs/emo/match.csv  chosen clips, similarity and emotion labels

Usage:  envs/seedvc/Scripts/python scripts/emotion_match.py [--min-score 0.6] [--k 3]
Round 4 (per phrase, see segment_lines.py):
        ... emotion_match.py --lines-csv data/source/phrases/phrases.csv --out data/default/refs/emo_phrase

--rule precise (2026-10-15): the raw cosine above agreed with the phrase's emotion2vec category 90 % of the time,
but its picks were barely closer than random clips in pitch, loudness and pace - the category was matched, the
intensity was not - and 9 categories are coarse (the game phrases came out 59 % "surprised", none calm or sad).
Each clip is scored on four terms, each standardized over the bank for that line, then weighted:
  emb     cosine of the emotion2vec embeddings with each corpus's mean removed (the game recording vs the anime
          clips differ as a whole; that offset said nothing about the emotion and squeezed every raw cosine
          into 0.98-0.997)
  cat     -Jensen-Shannon distance of the 9 emotion2vec class scores
  aro     -distance of the arousal cues (arousal_features.py: pitch level, pitch range, loudness, pace, vocal
          effort), each z-scored within its own corpus: "as excited as she gets" matches "as excited as the
          game character gets", whatever their registers
  type    sentence type from the text (question / exclamation / trailing off / statement) is the same
The best clip leads the reference; more clips (up to --k, --max-sec) join only with the same top category and a
score within MARGIN of the best, so one reference does not mix moods.
"""
import argparse
import csv
import re
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from vl_paths import LINES, ROOT, VOICE_DIR

ZT = VOICE_DIR  # the target voice (VL_VOICE)
OUT = ZT / "refs" / "emo"
SR = 44100
TOOLS_PY = ROOT / "envs" / "tools" / "Scripts" / "python.exe"  # parselmouth for the arousal cues
AROUSAL = {"pitch": 1.0, "range": 1.0, "level": 1.0, "rate": 0.5, "tilt": 1.0}  # cue -> weight in the distance
W_EMB, W_CAT, W_ARO, W_TYPE = 1.0, 1.0, 1.5, 0.5
MARGIN = 1.0  # a joined clip's score may trail the best by this much (the terms are in standard deviations)


def read_csv(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def clip_path(cid):
    """Prefer the de-reverbed / de-noised copy from clean_refs.py."""
    clean = ZT / "slices_clean" / f"{cid}.wav"
    return clean if clean.exists() else ZT / "slices" / f"{cid}.wav"


def bank(min_score, max_bg):
    from vl_paths import voice_cfg

    if voice_cfg().get("bank_from") == "train":  # the training set was picked another way (target_segments.py):
        return [r for r in read_csv(ZT / "train.csv") if float(r["duration"]) >= 1.2]  # its clips are the bank
    from vl_common import is_target_clip  # same purity rule as the training set

    rows = read_csv(ZT / "slices_ranked.csv")
    return [r for r in rows if is_target_clip(r, min_score=min_score, max_bg=max_bg, min_dur=1.2)]


def emotion_model():
    from funasr import AutoModel
    from huggingface_hub import snapshot_download

    local = ROOT / "cache" / "emotion2vec_plus_large"
    for attempt in range(5):  # 1.9 GB over a flaky link: resumable download, retried
        try:
            snapshot_download("emotion2vec/emotion2vec_plus_large", local_dir=str(local))
            break
        except Exception as e:  # noqa: BLE001
            print(f"download attempt {attempt + 1} failed: {e}")
    return AutoModel(model=str(local), disable_update=True)


def embed(model, path):
    """(unit embedding, top label, 9 class scores)"""
    wav, _ = librosa.load(str(path), sr=16000, mono=True)
    res = model.generate(wav, granularity="utterance", extract_embedding=True)[0]
    feat = np.asarray(res["feats"], dtype=np.float32).reshape(-1)
    label = res["labels"][int(np.argmax(res["scores"]))].split("/")[-1]
    return feat / (np.linalg.norm(feat) + 1e-9), label, np.asarray(res["scores"], dtype=np.float64)


def bank_embeddings(model, clips):
    """id -> (unit embedding, top label, class scores) for the bank, cached in data/<voice>/emo_bank.npz
    (keyed by clip id and file size + mtime; tts_testset.py reads the same cache)."""
    cache = ZT / "emo_bank.npz"
    have = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        for i, cid in enumerate(z["ids"]):
            have[str(cid)] = (str(z["stamps"][i]), z["feats"][i], str(z["labels"][i]), z["scores"][i])
    out, new = {}, 0
    for r in clips:
        p = clip_path(r["id"])
        stamp = f"{p.stat().st_size}-{p.stat().st_mtime_ns}"
        h = have.get(r["id"])
        if h and h[0] == stamp:
            out[r["id"]] = (h[1], h[2], h[3])
        else:
            out[r["id"]] = embed(model, p)
            new += 1
        have[r["id"]] = (stamp, *out[r["id"]])
    if new:
        ids = list(have)
        np.savez(cache, ids=np.array(ids), stamps=np.array([have[i][0] for i in ids]),
                 feats=np.stack([have[i][1] for i in ids]), labels=np.array([have[i][2] for i in ids]),
                 scores=np.stack([have[i][3] for i in ids]))
    print(f"emotion embeddings: {new} computed, {len(out) - new} from {cache.name}")
    return out


def arousal(items, out_csv):
    """items [(id, wav path, text)] -> id -> cue vector (AROUSAL order, NaN if missing); cached in out_csv."""
    lst = out_csv.with_name(out_csv.stem + "_list.csv")
    with open(lst, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "path", "text"])
        w.writerows(items)
    subprocess.run([str(TOOLS_PY), str(Path(__file__).with_name("arousal_features.py")), str(lst), str(out_csv)],
                   check=True, cwd=str(Path(__file__).parent))
    lst.unlink()
    num = lambda v: float(v) if v not in (None, "") else np.nan  # noqa: E731
    return {r["id"]: np.array([num(r[k]) for k in AROUSAL]) for r in read_csv(out_csv)}


def zscore(feats):
    ids = list(feats)
    m = np.array([feats[i] for i in ids])
    z = (m - np.nanmean(m, 0)) / (np.nanstd(m, 0) + 1e-9)
    return dict(zip(ids, z))


def sentence_type(text):
    t = re.sub(r"[」』）)\s]+$", "", (text or "").strip())
    if re.search(r"[?？]$", t):
        return "question"
    if re.search(r"[!！]$", t):
        return "exclaim"
    if re.search(r"(…|‥|\.\.\.|〜|~|ー|っ)$", t):
        return "trailing"
    return "statement"


def js(a, b):
    a, b = a / (a.sum() + 1e-12), b / (b.sum() + 1e-12)
    m = (a + b) / 2
    kl = lambda x, y: np.sum(x * np.log((x + 1e-12) / (y + 1e-12)), axis=-1)  # noqa: E731
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def std(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def precise_scores(e_line, s_line, a_line, typ, bank_e, bank_s, bank_a, bank_typ, mean_line, mean_bank):
    """Score of every bank clip for one line (see the module docstring); also the parts, for the report."""
    el = e_line - mean_line
    eb = bank_e - mean_bank
    emb = (eb @ el) / (np.linalg.norm(eb, axis=1) * np.linalg.norm(el) + 1e-9)
    cat = -js(bank_s, s_line[None])
    w = np.array(list(AROUSAL.values()))
    d = bank_a - a_line[None]
    ok = np.isfinite(d)
    aro = -np.sqrt(np.nansum(w * np.where(ok, d, 0) ** 2, 1) / np.maximum((w * ok).sum(1), 1e-9))
    aro = np.where(ok.any(1), aro, np.nanmin(aro))
    same = (bank_typ == typ).astype(float)
    total = W_EMB * std(emb) + W_CAT * std(cat) + W_ARO * std(aro) + W_TYPE * same
    return total, {"emb": emb, "cat": cat, "aro": -aro, "type": same}


def load(path, fade_ms=15):
    wav, _ = librosa.load(str(path), sr=SR, mono=True)
    wav = wav / (np.abs(wav).max() + 1e-9) * 0.9
    n = min(int(SR * fade_ms / 1000), len(wav) // 2)
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    wav[:n] *= ramp  # hard cuts at the joins are clicks the model would happily imitate
    wav[len(wav) - n:] *= ramp[::-1]
    return wav


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=0.6, help="see vl_common.is_target_clip")
    ap.add_argument("--k", type=int, default=3, help="max clips joined per reference")
    ap.add_argument("--max-sec", type=float, default=12.0)
    ap.add_argument("--max-bg", type=float, default=-10.0, help="max instrumental-minus-vocal level (dB)")
    ap.add_argument("--lines-csv", default=str(LINES / "lines.csv"), help="rows with id + text; <id>.wav beside it")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--emotions", default=None,
                    help="only target clips whose emotion2vec label is in this comma list, e.g. sad,fearful "
                         "(when the line's own label is wrong: L14's wounded grunt was tagged happy)")
    ap.add_argument("--rule", choices=["cosine", "precise"], default=None,
                    help="cosine: raw emotion2vec cosine (sets made before 2026-10-15); precise: see the docstring. "
                         "Default: the voice's emo_rule setting, else cosine")
    ap.add_argument("--min-sec", type=float, default=6.0, help="precise: keep joining same-category clips up to this")
    ap.add_argument("--bank-only", action="store_true",
                    help="only fill the bank's caches (emo_bank.npz, arousal_bank.csv) - for tts_testset.py")
    ap.add_argument("--embed-files", default=None,
                    help="only embed these wavs (a text file, one path per line) -> --embed-out npz (tts_eval.py)")
    ap.add_argument("--embed-out", default=None)
    a = ap.parse_args()
    if a.embed_files:
        model = emotion_model()
        paths = [ln.strip() for ln in Path(a.embed_files).read_text(encoding="utf-8").splitlines() if ln.strip()]
        res = [embed(model, p) for p in paths]
        np.savez(a.embed_out, paths=np.array(paths), feats=np.stack([r[0] for r in res]),
                 labels=np.array([r[1] for r in res]), scores=np.stack([r[2] for r in res]))
        raise SystemExit(0)
    lines_csv = Path(a.lines_csv)
    OUT = Path(a.out)
    from vl_paths import voice_cfg

    rule = a.rule or voice_cfg().get("emo_rule", "cosine")

    clips = bank(a.min_score, a.max_bg)
    n_clean = sum((ZT / "slices_clean" / f"{r['id']}.wav").exists() for r in clips)
    print(f"{VOICE_DIR.name} clip bank: {len(clips)} clips ({n_clean} cleaned); rule {rule}")
    model = emotion_model()
    clip_emb = bank_embeddings(model, clips)
    if a.bank_only:
        arousal([(r["id"], str(clip_path(r["id"])), r.get("text", "")) for r in clips], ZT / "arousal_bank.csv")
        raise SystemExit(0)
    lines = read_csv(lines_csv)
    if rule == "precise":
        OUT.mkdir(parents=True, exist_ok=True)
        mean_bank = np.mean([clip_emb[r["id"]][0] for r in clips], 0)  # before --emotions narrows the bank
        bank_ar = zscore(arousal([(r["id"], str(clip_path(r["id"])), r.get("text", "")) for r in clips],
                                 ZT / "arousal_bank.csv"))
        line_ar = zscore(arousal([(ln["id"], str(lines_csv.parent / f"{ln['id']}.wav"), ln.get("text", ""))
                                  for ln in lines], OUT / "arousal_lines.csv"))
    if a.emotions:
        keep = set(a.emotions.split(","))
        clips = [r for r in clips if clip_emb[r["id"]][1] in keep]
        print(f"restricted to {sorted(keep)}: {len(clips)} clips")

    OUT.mkdir(parents=True, exist_ok=True)
    gap = np.zeros(int(0.25 * SR), dtype=np.float32)
    line_emb = {ln["id"]: embed(model, lines_csv.parent / f"{ln['id']}.wav") for ln in lines}
    if rule == "precise":
        mean_line = np.mean([v[0] for v in line_emb.values()], 0)
        B_e = np.stack([clip_emb[r["id"]][0] for r in clips])
        B_s = np.stack([clip_emb[r["id"]][2] for r in clips])
        B_a = np.stack([bank_ar.get(r["id"], np.full(len(AROUSAL), np.nan)) for r in clips])
        B_t = np.array([sentence_type(r.get("text", "")) for r in clips])
    report = []
    for ln in lines:
        e, label, scores = line_emb[ln["id"]]
        extra = {}
        if rule == "precise":
            total, parts_ = precise_scores(e, scores, line_ar.get(ln["id"], np.full(len(AROUSAL), np.nan)),
                                           sentence_type(ln.get("text", "")), B_e, B_s, B_a, B_t, mean_line, mean_bank)
            order = np.argsort(-total)
            best = order[0]
            chosen_idx, sec = [], 0.0
            for j in order:
                d = float(clips[j]["duration"])
                if len(chosen_idx) >= a.k or sec + d > a.max_sec:
                    continue
                if chosen_idx and clip_emb[clips[j]["id"]][1] != clip_emb[clips[best]["id"]][1]:
                    continue  # one mood per reference
                if chosen_idx and total[j] < total[best] - MARGIN and sec >= a.min_sec:
                    break
                chosen_idx.append(j)
                sec += d + 0.25
            chosen = [clips[j] for j in chosen_idx]
            total_sec = sec
            extra = {"rule": rule, "line_type": sentence_type(ln.get("text", "")),
                     "scores": " ".join(f"{total[j]:.2f}" for j in chosen_idx),
                     "arousal_dist": " ".join(f"{parts_['aro'][j]:.2f}" for j in chosen_idx),
                     "ref_types": " ".join(str(B_t[j]) for j in chosen_idx),
                     "ref_ids": " | ".join(r["id"] for r in chosen)}
        else:
            ranked = sorted(clips, key=lambda r: -float(clip_emb[r["id"]][0] @ e))
            chosen, total_sec = [], 0.0
            for r in ranked:
                d = float(r["duration"])
                if len(chosen) >= a.k or total_sec + d > a.max_sec:
                    continue
                chosen.append(r)
                total_sec += d + 0.25
        parts = []
        for r in chosen:
            parts += [load(clip_path(r["id"])), gap]
        sf.write(OUT / f"{ln['id']}.wav", np.concatenate(parts[:-1]), SR, subtype="PCM_16")
        report.append({
            "id": ln["id"], "line_emotion": label, "line_text": ln["text"],
            "sims": " ".join(f"{float(clip_emb[r['id']][0] @ e):.3f}" for r in chosen),
            "ref_emotions": " ".join(clip_emb[r["id"]][1] for r in chosen),
            "ref_texts": " / ".join(r.get("text", "") for r in chosen),
            "ref_sec": round(total_sec - 0.25, 1), **extra,
        })
        print(f"{ln['id']} [{label}] -> {report[-1]['ref_emotions']}  ({report[-1]['ref_texts']})")
    with open(OUT / "match.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)
