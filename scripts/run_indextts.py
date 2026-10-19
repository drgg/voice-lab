"""Group D: IndexTTS-2.5 re-dub (run with the index-tts uv env).

For each line: text + target speaker prompt + the original line as emotion prompt.
Generates once at duration_factor=1, then rescales duration_factor so the speech span matches
the original line (up to --passes attempts, stop when within --tol).

Usage (after `. scripts/env.ps1`):  & $INDEXTTS_PY scripts/run_indextts.py [--smoke]
(the venv lives at $VL_CACHE\envs\indextts: kaldifst cannot open non-ASCII paths)

Round 4 --phrases: one emotion prompt per line flattened long lines, so generate per phrase
(segment_lines.py) with that phrase's own audio as emotion prompt, its own target reference
(--spk-dir refs/emo_phrase) and --cands sampled takes; select_phrases.py picks and stitches them.
  & $INDEXTTS_PY scripts/run_indextts.py --phrases data/source/phrases/phrases.csv --cands 4 \
      --spk-dir data/default/refs/emo_phrase --emo-alpha 1.0 --name D4_cands
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl_paths import EXP, LINES, ROOT, VOICE_DIR  # noqa: E402  (before the chdir below)

REPO = ROOT / "repos" / "index-tts"
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import librosa  # noqa: E402
import soundfile as sf  # noqa: E402

import torch  # noqa: E402

from indextts.infer_v2_5 import IndexTTS2  # noqa: E402


class PeakSafe(torch.nn.Module):
    """IndexTTS hard-clamps the vocoder output to int16 range, so any overshoot becomes clipping
    crackle. Scale the vocoder output under the ceiling before that clamp happens."""

    def __init__(self, inner, ceiling=0.97):
        super().__init__()
        self.inner, self.ceiling = inner, ceiling

    def forward(self, x):
        y = self.inner(x)
        peak = y.abs().max()
        return y * (self.ceiling / peak) if peak > self.ceiling else y


def span(path):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    _, (a, b) = librosa.effects.trim(wav, top_db=40)
    return (b - a) / sr


def lines():
    with open(LINES / "lines.csv", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            text = (r.get("text_edit") or "").strip() or r["text"].strip()
            yield r["id"], LINES / f"{r['id']}.wav", text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--spk", default=str(VOICE_DIR / "refs" / "ref_12s.wav"))
    ap.add_argument("--emo-alpha", type=float, default=0.7)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--spk-dir", default=None, help="per-line speaker prompts <spk-dir>/<id>.wav (overrides --spk)")
    ap.add_argument("--name", default="D_indextts25", help="output folder under exp/")
    ap.add_argument("--phrases", default=None, help="phrases.csv from segment_lines.py: candidates per phrase")
    ap.add_argument("--cands", type=int, default=4, help="sampled takes per phrase (with --phrases)")
    ap.add_argument("--force", action="store_true", help="--phrases: regenerate takes that could be reused")
    ap.add_argument("--more", default="", help="--phrases: phrase ids that get --more-cands takes (none fitted)")
    ap.add_argument("--more-cands", type=int, default=10, help="takes for the --more phrases (existing ones kept)")
    args = ap.parse_args()
    OUT = EXP / args.name

    tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_bf16=True, use_cuda_kernel=False)
    tts.bigvgan = PeakSafe(tts.bigvgan)

    if args.smoke:
        out = ROOT / "logs" / "smoke_indextts.wav"
        spk = ROOT / "repos/seed-vc/examples/reference/azuma_0.wav"
        t = time.time()
        tts.infer(spk_audio_prompt=str(spk), text="こんにちは、今日はいい天気ですね。", lang="ja", output_path=str(out))
        print(f"smoke ok: {span(out):.2f}s speech in {time.time() - t:.1f}s -> {out}")
        sys.exit(0)

    OUT.mkdir(parents=True, exist_ok=True)
    if args.phrases:
        phrases_csv = Path(args.phrases)
        with open(phrases_csv, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        prev = {}  # takes of an earlier run, reused when the phrase is unchanged (same text and length)
        if (OUT / "cands.json").exists() and not args.force:
            prev = {p["id"]: p for p in json.loads((OUT / "cands.json").read_text(encoding="utf-8"))["phrases"]}
        log = []
        more = set(filter(None, args.more.split(",")))
        for r in rows:
            src = phrases_csv.parent / f"{r['id']}.wav"
            target = float(r["speech_end"]) - float(r["speech_start"])
            want = max(args.cands, args.more_cands) if r["id"] in more else args.cands
            old = prev.get(r["id"])
            keep = old["cands"] if (old and old["text"] == r["text"] and abs(old["target_span"] - target) < 0.02
                                    and all((OUT / c["file"]).exists() for c in old["cands"])) else []
            if len(keep) >= want:
                log.append(old)
                continue
            spk = str(Path(args.spk_dir) / f"{r['id']}.wav") if args.spk_dir else args.spk
            # takes already made stay; new ones continue the seed sequence and the first take's length aim
            cands, t0 = list(keep), time.time()
            factor = min(2.0, max(0.5, target / max(cands[0]["span"], 1e-3))) if cands else 1.0
            for k in range(len(cands), want):
                torch.manual_seed(1000 + k)  # a different sampled take per candidate, reproducible
                out = OUT / f"{r['id']}_c{k}.wav"
                tts.infer(spk_audio_prompt=spk, text=r["text"], lang="ja", output_path=str(out),
                          emo_audio_prompt=str(src), emo_alpha=args.emo_alpha, duration_factor=factor)
                got = span(out)
                cands.append({"file": out.name, "factor": round(factor, 3), "span": round(got, 3)})
                if k == 0:  # later takes aim at the phrase length measured on the first one
                    factor = min(2.0, max(0.5, target / max(got, 1e-3)))
            log.append({"id": r["id"], "text": r["text"], "target_span": round(target, 3), "cands": cands,
                        "seconds": round(time.time() - t0, 2)})
            print(f"[{args.name}] {r['id']} target {target:.2f}s spans {[c['span'] for c in cands]}")
        params = {"spk": args.spk_dir or args.spk, "emo_alpha": args.emo_alpha, "cands": args.cands, "bf16": True}
        (OUT / "cands.json").write_text(json.dumps({"params": params, "phrases": log}, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
        sys.exit(0)

    log = []
    for lid, src, text in lines():
        target = span(src)
        spk = str(Path(args.spk_dir) / f"{lid}.wav") if args.spk_dir else args.spk
        factor, t0, tries = 1.0, time.time(), []
        for k in range(args.passes):
            tmp = OUT / f"_{lid}_try{k}.wav"
            tts.infer(spk_audio_prompt=spk, text=text, lang="ja", output_path=str(tmp),
                      emo_audio_prompt=str(src), emo_alpha=args.emo_alpha, duration_factor=factor)
            got = span(tmp)
            tries.append({"factor": round(factor, 3), "span": round(got, 3), "file": tmp})
            if abs(got - target) / target <= args.tol:
                break
            factor = min(2.0, max(0.5, factor * target / max(got, 1e-3)))
        # sampling is random, so keep the closest attempt's own file rather than re-rendering it
        best = min(tries, key=lambda x: abs(x["span"] - target))
        best["file"].replace(OUT / f"{lid}.wav")
        for t in tries:
            t["file"].unlink(missing_ok=True)
            del t["file"]
        log.append({"id": lid, "text": text, "target_span": round(target, 3), "tries": tries,
                    "chosen": {"factor": best["factor"], "span": best["span"]},
                    "seconds": round(time.time() - t0, 2)})
        print(f"[{args.name}] {lid} target {target:.2f}s tries {tries}")
    params = {"spk": args.spk_dir or args.spk, "emo_alpha": args.emo_alpha, "passes": args.passes,
              "tol": args.tol, "bf16": True}
    (OUT / "run.json").write_text(json.dumps({"params": params, "lines": log}, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
