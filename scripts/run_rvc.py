"""RVC (Applio) for the target voice (VL_VOICE): train on data/<voice>/train, then convert experiment inputs.

Applio lives at $VL_CACHE\\Applio with its venv at $VL_CACHE\\envs\\applio
(ASCII path; Python 3.12 for its pins). Its functions resolve rvc/ and logs/ relative to the repo,
so this script chdirs there and calls core.py's functions directly.

  train   prerequisites -> preprocess (40 kHz) -> extract (RMVPE + contentvec) -> train -> index
  export  rebuild the inference weights from the latest G_ checkpoint (if training's export failed)
  infer   convert a folder of lines with the trained model:
            --src source   original lines of the current set -> exp/E_rvc   (--route nonverbal: only those)
            --src <group>  another group's 22 kHz output     -> exp/<group>_rvc   (two-stage polish)
            --src <dir>    any folder of L*.wav (refine phrases)

Usage:  $VL_CACHE\\envs\\applio\\Scripts\\python scripts/run_rvc.py train [--epochs 300]
        ... scripts/run_rvc.py infer --src D3_indextts_emo [--index-rate 0.5]
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl_paths import EXP, LINES, ROOT, VOICE, VOICE_DIR, routes, voice_cfg  # noqa: E402  (before the chdir below)

APPLIO = Path(os.environ.get("VL_CACHE", r"C:\voice-lab-cache")) / "Applio"
MODEL = voice_cfg().get("rvc_model") or f"{VOICE}_v1"  # voice.json rvc_model: a retrained set gets a new name
SR = 40000
os.chdir(APPLIO)
sys.path.insert(0, str(APPLIO))

from core import (  # noqa: E402
    run_extract_script, run_infer_script, run_prerequisites_script, run_preprocess_script, run_train_script,
)


def ensure_assets_config():
    """Applio's UI creates assets/config.json from the template on first launch; without it every
    in-training export of the inference weights fails (only the G_/D_ checkpoints get saved)."""
    cfg = APPLIO / "assets" / "config.json"
    if not cfg.exists():
        shutil.copyfile(APPLIO / "assets" / "config_template.json", cfg)


def export(a):
    """Build the inference .pth (<model>_<epoch>e_<step>s.pth) from the latest G_ checkpoint."""
    import torch
    from rvc.train.process.extract_model import extract_model
    from rvc.train.utils import HParams, latest_checkpoint_path

    ensure_assets_config()
    exp = APPLIO / "logs" / MODEL
    g = latest_checkpoint_path(str(exp), "G_*.pth")
    ck = torch.load(g, map_location="cpu", weights_only=True)
    hps = HParams(**json.loads((exp / "config.json").read_text(encoding="utf-8")))
    epoch, step = ck["iteration"], int(Path(g).stem.split("_")[1])  # save_checkpoint stores the epoch
    extract_model(ckpt=ck["model"], sr=hps.data.sample_rate, name=MODEL, epoch=epoch, step=step, hps=hps,
                  model_path=str(exp / f"{MODEL}_{epoch}e_{step}s.pth"), vocoder="HiFi-GAN")


def train(a):
    ensure_assets_config()
    # Applio defaults to 4 spawned loader workers (prefetch 8): with 16 GB RAM the commit charge grew the
    # C: pagefile until the disk was nearly full. The 7-min set fits in VRAM, so load it once, no workers.
    os.environ["RVC_NUM_WORKERS"] = "0"  # read by the patched rvc/train/train.py
    run_prerequisites_script(pretraineds_hifigan=True, models=True, exe=True)
    procs = 4  # one process per core (12) briefly flooded RAM; 7 minutes of audio needs no more
    print(run_preprocess_script(MODEL, str(VOICE_DIR / "train"), SR, procs,
                                "Automatic", False, False, 0.7, 3.0, 0.3, "none"))
    print(run_extract_script(MODEL, "rmvpe", procs, 0, SR, "contentvec", None, 2))
    print(run_train_script(MODEL, save_every_epoch=50, save_only_latest=False, save_every_weights=True,
                           total_epoch=a.epochs, sample_rate=SR, batch_size=a.batch, gpu=0, pretrained=True,
                           cleanup=False, index_algorithm="Auto", cache_data_in_gpu=True, vocoder="HiFi-GAN"))


def latest_weights():
    """Applio writes <model>_<epoch>e_<step>s.pth into logs/<model>; take the highest epoch."""
    pths = glob.glob(str(APPLIO / "logs" / MODEL / f"{MODEL}_*e_*s.pth"))
    if not pths:
        raise SystemExit("no trained weights found - run `train` first")
    pth = max(pths, key=lambda p: int(Path(p).stem.split("_")[-2].rstrip("e")))
    index = next(iter(glob.glob(str(APPLIO / "logs" / MODEL / "*.index"))), "")
    return pth, index


def infer(a):
    pth, index = latest_weights()
    if a.src == "source":  # the current set's original lines (vl_paths)
        src_dir, out_dir = LINES, EXP / "E_rvc"
    elif Path(a.src).is_dir():  # any folder of L*.wav (refine phrases); needs --name
        src_dir, out_dir = Path(a.src), EXP / (a.name or Path(a.src).name + "_rvc")
    else:
        src_dir, out_dir = EXP / a.src, EXP / f"{a.src}_rvc"
    if a.name:
        out_dir = EXP / a.name
    if a.out:  # outside the set's exp folder (e.g. a text-to-speech test set's T01.wav ...)
        out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(src_dir.glob(a.pattern))
    if a.route:  # e.g. only the non-verbal lines (scan_lines.py) for the production run
        keep = {k for k, v in routes().items() if v in a.route.split(",")}
        sources = [s for s in sources if s.stem in keep]
    if a.lines:  # e.g. the lines rvc_guard.py found sung off, redone with another pitch tracker
        keep = set(a.lines.split(","))
        sources = [s for s in sources if s.stem in keep]
    import numpy as np
    import soundfile as sf

    log = []
    for src in sources:
        dst = out_dir / src.name
        if not a.force and dst.exists() and dst.stat().st_mtime > src.stat().st_mtime:
            continue  # up to date: keeps the 48 kHz copy made from it valid too
        t = time.time()
        inp, outp = src, dst
        if a.pad > 0:
            # a 0.1-0.5 s grunt gives the pitch tracker and content encoder no context: RMVPE misses
            # the voicing (the grunt comes out as hiss) or jumps an octave. Pad with silence, trim after.
            x, sr = sf.read(str(src), dtype="float32", always_2d=True)
            z = np.zeros((int(a.pad * sr), x.shape[1]), dtype=np.float32)
            inp, outp = out_dir / f"_pad_{src.name}", out_dir / f"_out_{src.name}"
            sf.write(str(inp), np.concatenate([z, x, z]), sr)
        run_infer_script(
            pitch=a.pitch, index_rate=a.index_rate, volume_envelope=a.volume_envelope, protect=a.protect,
            f0_method=a.f0_method, input_path=str(inp), output_path=str(outp), pth_path=pth,
            index_path=index, split_audio=False, f0_autotune=False, f0_autotune_strength=1.0,
            proposed_pitch=False, proposed_pitch_threshold=155.0, clean_audio=False, clean_strength=0.5,
            export_format="WAV", embedder_model="contentvec",
        )
        if a.pad > 0:
            y, sr = sf.read(str(outp), dtype="float32")
            n = int(a.pad * sr)
            sf.write(str(dst), y[n:len(y) - n], sr, subtype="PCM_16")
            inp.unlink()
            outp.unlink()
        log.append({"id": src.stem, "seconds": round(time.time() - t, 2)})
        print(f"[{out_dir.name}] {src.stem} {log[-1]['seconds']}s")
    params = {"model": Path(pth).name, "index": Path(index).name if index else "", "pitch": a.pitch,
              "index_rate": a.index_rate, "protect": a.protect, "volume_envelope": a.volume_envelope,
              "f0_method": a.f0_method, "pad": a.pad}
    (out_dir / "run.json").write_text(json.dumps({"params": params, "lines": log}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "export", "infer"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--src", default="source")
    ap.add_argument("--name", default=None, help="output folder under exp/ (default E_rvc / <src>_rvc)")
    ap.add_argument("--route", default=None, help="only lines with this scan_lines.py route, e.g. nonverbal")
    ap.add_argument("--pitch", type=int, default=0, help="semitones; 0 for same-actress conversion")
    ap.add_argument("--index-rate", type=float, default=0.5)
    ap.add_argument("--protect", type=float, default=0.33)
    ap.add_argument("--volume-envelope", type=float, default=1.0)
    ap.add_argument("--f0-method", default="rmvpe")
    ap.add_argument("--pad", type=float, default=0.0, help="seconds of silence around each input (short grunts)")
    ap.add_argument("--force", action="store_true", help="redo outputs that are newer than their source")
    ap.add_argument("--lines", default="", help="only these line ids (comma list)")
    ap.add_argument("--pattern", default="L*.wav", help="input files in --src (T*.wav for tts_testset.py's keys)")
    ap.add_argument("--out", default="", help="output folder instead of exp/<name>")
    a = ap.parse_args()
    {"train": train, "export": export, "infer": infer}[a.cmd](a)
