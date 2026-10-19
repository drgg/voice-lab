"""Score a text-to-speech system on the voice's held-out test set (tools env; VL_VOICE).

  python scripts/tts_eval.py <gen dir> --name <system> [--page]

<gen dir> holds T01.wav ... (tts_testset.py keys), each spoken from that clip's text alone. Per clip, against her
real clip:
  timbre   ECAPA cosine to her training centroid (and to the real clip)
  words    CER of a Whisper transcript against the text
  emotion  emotion2vec's top category equals the real clip's; cosine of the embeddings with her bank's mean removed
           (seedvc env, through emotion_match.py --embed-files)
  arousal  distance of the arousal cues (arousal_features.py; z-scored with her bank's spread; loudness left out:
           the output gain is arbitrary) to the real clip's
  melody   contour correlation with the real clip (DTW-aligned), median pitch offset (st), range ratio, length ratio
  quality  SQUIM's reference-free PESQ / SI-SDR estimates
The same measures for each clip's style reference - another real clip of hers in a similar mood - give the scale
("ref" row): emotion, arousal and melody are judged against what a real clip of hers that was not this one scores.
Results: data/<voice>/tts/eval/<name>.csv and summary.json (every system scored so far); STANDARD below is checked.
--page: a blind listening page eval/<voice>_tts/<name>/ (real vs generated, which is real; ratings).
"""
import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import prosody
from arousal_features import features as arousal_raw
from emotion_match import AROUSAL, clip_path
from vl_common import EVAL, ROOT, TARGET, normalize_ja, read_csv, read_mono, speech_span, transcribe, \
    whisper_model, write_csv
from vl_paths import voice_cfg

SEEDVC_PY = ROOT / "envs" / "seedvc" / "Scripts" / "python.exe"
os.environ.setdefault("TORCH_HOME", r"E:\voice-lab-cache\torch")
# what "good enough to dub the game lines" means (agreed direction 2026-10-15; the user's listening test decides)
STANDARD = [
    ("timbre", "ECAPA to her centroid (median) >= 95 % of her real clips'", lambda s, r, f: s["timbre"] >= 0.95 * r["timbre"]),
    ("cer", "CER (median) <= 0.10 and <= real + 0.05", lambda s, r, f: s["cer"] <= max(0.10, r["cer"] + 0.05)),
    ("emotion", "same emotion2vec category as the real clip >= reference's rate - 0.10",
     lambda s, r, f: s["emotion_same"] >= f["emotion_same"] - 0.10),
    ("arousal", "arousal distance to the real clip (median) <= reference's + 0.10",
     lambda s, r, f: s["arousal_dist"] <= f["arousal_dist"] + 0.10),
    ("quality", "SQUIM PESQ (median) >= real - 0.3", lambda s, r, f: s["pesq"] >= r["pesq"] - 0.3),
]


def cer(ref, hyp):
    import jiwer

    r = normalize_ja(ref)
    return float(jiwer.cer(r, normalize_ja(hyp))) if r else 0.0


def emotions(paths, work):
    lst = work / "emo_list.txt"
    lst.write_text("\n".join(str(p) for p in paths), encoding="utf-8")
    subprocess.run([str(SEEDVC_PY), str(Path(__file__).with_name("emotion_match.py")), "--embed-files", str(lst),
                    "--embed-out", str(work / "emo.npz")], check=True, cwd=str(Path(__file__).parent),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    z = np.load(work / "emo.npz")
    return {str(p): (f, str(lb)) for p, f, lb in zip(z["paths"], z["feats"], z["labels"])}


def score(pairs, name, work):
    """pairs: [(key, generated path, real path, text)] -> rows of per-clip measures."""
    from prep_voice import ecapa, embed
    import torch
    import torchaudio

    spk = ecapa()
    hold = set((TARGET / "tts" / "holdout_ids.txt").read_text(encoding="utf-8").split("\n"))
    train = [r for r in read_csv(TARGET / "train.csv") if r["id"] not in hold]
    cen = np.mean([embed(spk, clip_path(r["id"])) for r in random.Random(1).sample(train, min(300, len(train)))], 0)
    cen /= np.linalg.norm(cen)
    asr = whisper_model()
    squim = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().eval()
    bank = np.load(TARGET / "emo_bank.npz")
    mean_bank = bank["feats"].mean(0)
    ab = read_csv(TARGET / "arousal_bank.csv")
    num = lambda v: float(v) if v not in ("", None) else np.nan  # noqa: E731
    A = np.array([[num(r[k]) for k in AROUSAL] for r in ab])
    mu, sd = np.nanmean(A, 0), np.nanstd(A, 0) + 1e-9
    # loudness is left out: a generated file's level is the model's output gain, not how loud she speaks
    # (gsv_base came out 6.4 of her bank's sd quieter than the real clips, swamping every other cue)
    w = np.array([0.0 if k == "level" else v for k, v in AROUSAL.items()])
    emo = emotions([p for _, g, r, _ in pairs for p in (g, r)], work)
    rows = []
    for key, g, r, text in pairs:
        xg, sg = read_mono(g)
        xr, sr = read_mono(r)
        eg, er = embed(spk, g), embed(spk, r)
        hyp, _, _ = transcribe(asr, g)
        y = torch.from_numpy(np.ascontiguousarray(prosody._resample(xg, sg), dtype=np.float32))[None]
        with torch.inference_mode():
            _, pesq, sisdr = squim(y)
        ag, ar = arousal_raw(g, text), arousal_raw(r, text)
        if ag and ar:
            d = (np.array([ag[k] if ag[k] != "" else np.nan for k in AROUSAL]) - np.array([ar[k] if ar[k] != "" else np.nan for k in AROUSAL])) / sd
            ok = np.isfinite(d)
            adist = float(np.sqrt(np.sum(w[ok] * d[ok] ** 2) / max(w[ok].sum(), 1e-9)))
        else:
            adist = np.nan
        fg, lg = emo[str(g)]
        fr, lr = emo[str(r)]
        cg, cr = fg - mean_bank, fr - mean_bank
        rows.append({
            "key": key, "system": name, "text": text, "asr": hyp,
            "timbre": round(float(eg @ cen), 3), "timbre_to_real": round(float(eg @ er), 3),
            "cer": round(cer(text, hyp), 3),
            "emotion": lg, "emotion_real": lr, "emotion_same": int(lg == lr),
            "emotion_cos": round(float(cg @ cr / (np.linalg.norm(cg) * np.linalg.norm(cr) + 1e-9)), 3),
            "arousal_dist": round(adist, 3),
            "contour_corr": round(prosody.contour_corr(xr, sr, xg, sg), 3),
            "pitch_offset": round(prosody.median_st(xg, sg) - prosody.median_st(xr, sr), 2),
            "range_ratio": round(prosody.range_ratio(xr, sr, xg, sg), 2),
            "length_ratio": round(speech_span(xg, sg) / max(speech_span(xr, sr), 1e-3), 2),
            "pesq": round(float(pesq), 2), "si_sdr": round(float(sisdr), 1),
        })
    return rows


def excluded():
    """Test lines left out of the summaries: tts/test_exclude.txt (key<TAB>reason), e.g. a clip holding another
    voice (clip_purity.py) or a wrong transcript (transcript_check.py)."""
    p = TARGET / "tts" / "test_exclude.txt"
    return {ln.split("\t")[0] for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()} if p.exists() else set()


def summarize(rows):
    skip = excluded()
    rows = [r for r in rows if r["key"] not in skip]
    med = lambda k: float(np.nanmedian([float(r[k]) for r in rows if r[k] not in ("", None)]))  # noqa: E731
    return {"n": len(rows), "timbre": med("timbre"), "timbre_to_real": med("timbre_to_real"), "cer": med("cer"),
            "emotion_same": float(np.mean([float(r["emotion_same"]) for r in rows])), "emotion_cos": med("emotion_cos"),
            "arousal_dist": med("arousal_dist"), "contour_corr": med("contour_corr"),
            "pitch_offset": med("pitch_offset"), "range_ratio": med("range_ratio"), "length_ratio": med("length_ratio"),
            "pesq": med("pesq"), "si_sdr": med("si_sdr")}


def page(test, gen_dir, name):
    out = EVAL.parent / f"{TARGET.name}_tts" / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    rng = random.Random(name)
    items = []
    for t in test:
        g = gen_dir / f"{t['key']}.wav"
        if not g.exists():
            continue
        opts = [("real", clip_path(t["id"])), ("gen", g)]
        rng.shuffle(opts)
        o = []
        for j, (lab, p) in enumerate(opts):
            x, sr = sf.read(str(p), dtype="float32", always_2d=True)
            x = x.mean(1)
            x = x * (10 ** (-20 / 20) / (np.sqrt(np.mean(x ** 2)) + 1e-9))  # same loudness
            nm = f"{t['key']}_{'xy'[j]}.wav"
            sf.write(out / nm, np.clip(x, -0.99, 0.99), sr, subtype="PCM_16")
            o.append({"src": nm, "kind": lab})
        items.append({"key": t["key"], "text": t["text"], "opts": o})
    tpl = (Path(__file__).parent / "templates" / "tts_blind.html").read_text(encoding="utf-8")
    (out / "index.html").write_text(tpl.replace("__DATA__", json.dumps(
        {"voice": voice_cfg()["label"], "system": name, "items": items}, ensure_ascii=False)), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_dir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--page", action="store_true")
    ap.add_argument("--resummarize", action="store_true",
                    help="only recompute summary.json from the per-clip csvs (after test_exclude.txt changed)")
    a = ap.parse_args()
    if a.resummarize:
        ev = TARGET / "tts" / "eval"
        summ = {p.stem: summarize(read_csv(p)) for p in sorted(ev.glob("*.csv"))}
        (ev / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
        for s, v in summ.items():
            print(f"{s:22} n {v['n']:3}  timbre {v['timbre']:.3f}  to_real {v['timbre_to_real']:.3f}  emo {v['emotion_same']:.2f}  "
                  f"arousal {v['arousal_dist']:.2f}  contour {v['contour_corr']:.2f}  range {v['range_ratio']:.2f}  pesq {v['pesq']:.2f}")
        raise SystemExit(0)
    gen_dir = Path(a.gen_dir)
    test = read_csv(TARGET / "tts" / "testset.csv")
    ev = TARGET / "tts" / "eval"
    work = ev / f"_work_{a.name}"
    work.mkdir(parents=True, exist_ok=True)
    summ_p = ev / "summary.json"
    summ = json.loads(summ_p.read_text(encoding="utf-8")) if summ_p.exists() else {}
    if "real" not in summ or "ref" not in summ:  # her real clips; her style references read against them
        real_rows = score([(t["key"], clip_path(t["id"]), clip_path(t["id"]), t["text"]) for t in test], "real", work)
        ref_rows = score([(t["key"], clip_path(t["ref_id"]), clip_path(t["id"]), t["text"]) for t in test], "ref", work)
        for r in ref_rows:
            r["cer"] = ""  # the reference says other words
        write_csv(ev / "real.csv", real_rows)
        write_csv(ev / "ref.csv", ref_rows)
        summ["real"], summ["ref"] = summarize(real_rows), summarize(ref_rows)
    pairs = [(t["key"], gen_dir / f"{t['key']}.wav", clip_path(t["id"]), t["text"]) for t in test
             if (gen_dir / f"{t['key']}.wav").exists()]
    rows = score(pairs, a.name, work)
    write_csv(ev / f"{a.name}.csv", rows)
    summ[a.name] = summarize(rows)
    summ_p.write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    keys = ["n", "timbre", "timbre_to_real", "cer", "emotion_same", "emotion_cos", "arousal_dist", "contour_corr",
            "pitch_offset", "range_ratio", "length_ratio", "pesq", "si_sdr"]
    print(f"{'':12}" + "".join(f"{k[:12]:>13}" for k in keys))
    for s in ("real", "ref", a.name):
        print(f"{s:12}" + "".join(f"{summ[s][k]:13.3f}" if isinstance(summ[s][k], float) else f"{summ[s][k]:13}" for k in keys))
    print("\nstandard:")
    for k, what, test_fn in STANDARD:
        print(f"  {'PASS' if test_fn(summ[a.name], summ['real'], summ['ref']) else 'fail'}  {what}")
    if a.page:
        print(f"page: {page(test, gen_dir, a.name)}")
