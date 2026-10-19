"""Groups A/B: Seed-VC V2, timbre-only and with style conversion (run in envs/seedvc).

Loads the V2 wrapper once and converts every line in data/source/lines/ against the target reference.
  A_seedvc_timbre  convert_style=False
  B_seedvc_style   convert_style=True  (AR model also transfers accent/emotion/speaking style)

Usage:  envs/seedvc/Scripts/python scripts/run_seedvc.py [--smoke] [--groups A,B] [--steps 30]
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

REPO = ROOT / "repos" / "seed-vc"
os.chdir(REPO)  # the wrapper resolves configs/ and ./checkpoints relative to the repo
sys.path.insert(0, str(REPO))

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

DEVICE = torch.device("cuda")
DTYPE = torch.float16
GROUPS = {"A": ("A_seedvc_timbre", False), "B": ("B_seedvc_style", True)}


def load_wrapper(cfm_ckpt=None, ar_ckpt=None):
    """None loads the released V2 weights; pass fine-tuned runs/<name>/CFM_*.pth / AR_*.pth to override."""
    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    vc = instantiate(cfg)
    vc.load_checkpoints(cfm_checkpoint_path=cfm_ckpt, ar_checkpoint_path=ar_ckpt)
    vc.to(DEVICE)
    vc.eval()
    vc.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=DTYPE, device=DEVICE)
    return vc


def peak_safe(audio, ceiling=0.97):
    """The decoder can overshoot +-1.0; writing that to 16-bit PCM clips it into audible crackle."""
    import numpy as np

    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    return audio * (ceiling / peak) if peak > ceiling else audio


def convert(vc, src, ref, style, args):
    full = None
    for _, full in vc.convert_voice_with_streaming(
        source_audio_path=str(src), target_audio_path=str(ref),
        diffusion_steps=args.steps, length_adjust=1.0,
        intelligebility_cfg_rate=args.int_cfg, similarity_cfg_rate=args.sim_cfg,
        top_p=0.9, temperature=1.0, repetition_penalty=1.0,
        convert_style=style, anonymization_only=False,
        device=DEVICE, dtype=DTYPE, stream_output=True,
    ):
        pass
    return full  # (sr, np.ndarray)


def lines():
    with open(LINES / "lines.csv", newline="", encoding="utf-8-sig") as f:
        return [(r["id"], LINES / f"{r['id']}.wav") for r in csv.DictReader(f)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="convert one repo example to check the setup")
    ap.add_argument("--groups", default="A,B")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--int-cfg", type=float, default=0.7)
    ap.add_argument("--sim-cfg", type=float, default=0.7)
    ap.add_argument("--ref", default=str(VOICE_DIR / "refs" / "ref_25s.wav"))
    ap.add_argument("--ref-dir", default=None, help="per-line references <ref-dir>/<id>.wav (overrides --ref)")
    ap.add_argument("--name", default=None, help="output folder under exp/ (single group only)")
    ap.add_argument("--src-dir", default=None, help="convert every <src-dir>/*.wav instead of the source lines")
    ap.add_argument("--cfm-ckpt", default=None, help="fine-tuned CFM checkpoint (timbre)")
    ap.add_argument("--ar-ckpt", default=None, help="fine-tuned AR checkpoint (style, used with convert_style)")
    args = ap.parse_args()

    vc = load_wrapper(args.cfm_ckpt, args.ar_ckpt)
    if args.smoke:
        src = REPO / "examples/source/glados_0.wav"
        ref = REPO / "examples/reference/azuma_0.wav"
        out = ROOT / "logs" / "smoke_seedvc.wav"
        t = time.time()
        sr, audio = convert(vc, src, ref, True, args)
        sf.write(out, audio, sr)
        print(f"smoke ok: {src.name} -> {ref.name}, {len(audio) / sr:.2f}s audio in {time.time() - t:.1f}s -> {out}")
        sys.exit(0)

    for g in args.groups.split(","):
        name, style = GROUPS[g.strip()]
        out_dir = EXP / (args.name or name)
        out_dir.mkdir(parents=True, exist_ok=True)
        log = []
        sources = [(p.stem, p) for p in sorted(Path(args.src_dir).glob("*.wav"))] if args.src_dir else lines()
        for lid, src in sources:
            t = time.time()
            ref = Path(args.ref_dir) / f"{lid}.wav" if args.ref_dir else args.ref
            sr, audio = convert(vc, src, ref, style, args)
            sf.write(out_dir / f"{lid}.wav", peak_safe(audio), sr)
            log.append({"id": lid, "seconds": round(time.time() - t, 2), "out_sec": round(len(audio) / sr, 2)})
            print(f"[{name}] {lid} {log[-1]}")
        params = {"steps": args.steps, "int_cfg": args.int_cfg, "sim_cfg": args.sim_cfg,
                  "convert_style": style, "ref": str(args.ref_dir or args.ref),
                  "cfm_ckpt": args.cfm_ckpt, "ar_ckpt": args.ar_ckpt}
        (out_dir / "run.json").write_text(json.dumps({"params": params, "lines": log}, indent=2), encoding="utf-8")
