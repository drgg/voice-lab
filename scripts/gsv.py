"""GPT-SoVITS: a text-to-speech model fine-tuned on the target voice (VL_VOICE) - prepare, train, speak.

Lives at $VL_CACHE\\GPT-SoVITS with its env $VL_CACHE\\envs\\gptsovits (Python 3.11, torch 2.7 cu128), both on an
ASCII path: pyopenjtalk's dictionary and some audio loaders fail on others. There is no C compiler on this machine,
so jieba_fast is a shim over the pure-Python jieba (Chinese only) and pyopenjtalk comes as pyopenjtalk-plus wheels.
Version v2ProPlus: by its authors' notes it does best with average-quality training sets and follows the training
set more than the prompt (v3/v4 lean on the prompt) - here: clips cut from longer recordings, and the speaker's style is what it should learn.

  prepare  data/<voice>/train.csv minus tts/holdout_ids.txt -> <name>: wavs copied under the ASCII tree, list file
           (wav|speaker|ja|text), then GPT-SoVITS's own steps 1-get-text, 2-get-hubert-wav32k, 2-get-sv,
           3-get-semantic -> $GSV/logs/<name>
  train    SoVITS (s2; --sovits-epochs, default 8) then GPT (s1; --gpt-epochs, default 15); batch 8 each - the
           WebUI's defaults for a 16 GB card
  speak    --list csv (key,text,ref_path,ref_text) -> --out/<key>.wav, with <name>'s latest weights (or --base:
           the pretrained models, no fine-tuning). Prompts are copied to the ASCII tree first.

  $VL_CACHE\\envs\\gptsovits\\Scripts\\python scripts/gsv.py prepare --name myvoice_gsv1     (VL_VOICE=myvoice)
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl_paths import ROOT, VOICE, VOICE_DIR  # noqa: E402

CACHE = Path(os.environ.get("VL_CACHE", r"C:\voice-lab-cache"))
GSV = CACHE / "GPT-SoVITS"
DATA = CACHE / "gptsovits-data"
VERSION = "v2ProPlus"
FFMPEG = ROOT / "tools" / "ffmpeg" / "bin"
PRE = "GPT_SoVITS/pretrained_models"
BASE = {"bert": f"{PRE}/chinese-roberta-wwm-ext-large", "hubert": f"{PRE}/chinese-hubert-base",
        "sv": f"{PRE}/sv/pretrained_eres2netv2w24s4ep4.ckpt", "s2G": f"{PRE}/v2Pro/s2Gv2ProPlus.pth",
        "s2D": f"{PRE}/v2Pro/s2Dv2ProPlus.pth", "s1": f"{PRE}/s1v3.ckpt"}


def read_csv(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def clean_text(t):
    """Whisper marks a pause inside a clip with a space; GPT-SoVITS reads a comma as one."""
    t = re.sub(r"[ 　]+", "、", t.strip())
    return re.sub(r"、+([。！？!?、])", r"\1", t)


def gsv_env(**extra):
    """The environment GPT-SoVITS's scripts expect: its root and GPT_SoVITS/ importable (the WebUI does it)."""
    e = dict(os.environ, PYTHONPATH=os.pathsep.join([str(GSV), str(GSV / "GPT_SoVITS")]),
             PATH=str(FFMPEG) + os.pathsep + os.environ.get("PATH", ""))  # its loaders call ffmpeg
    e.update({k: str(v) for k, v in extra.items()})
    return e


def run_step(script, env):
    e = gsv_env(**env)
    print(f"--- {script}", flush=True)
    subprocess.run([sys.executable, "-s", f"GPT_SoVITS/{script}"], cwd=str(GSV), env=e, check=True)


def prepare(a):
    hold = set((VOICE_DIR / "tts" / "holdout_ids.txt").read_text(encoding="utf-8").splitlines())  # ids hold spaces
    if a.exclude_flagged:
        from vl_paths import flagged_clips

        hold |= flagged_clips()
    rows = [r for r in read_csv(VOICE_DIR / "train.csv") if r["id"] not in hold and r["text"].strip()]
    print(f"{len(rows)} training clips ({len(hold)} held out or flagged)", flush=True)
    d = DATA / a.name
    wavs = d / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in rows:
        src = VOICE_DIR / "train" / r["file"]
        dst = wavs / r["file"]
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        lines.append(f"{r['file']}|{VOICE}|ja|{clean_text(r['text'])}")
    lst = d / f"{a.name}.list"
    lst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} clips, {sum(float(r['duration']) for r in rows) / 60:.1f} min -> {lst}")
    opt = GSV / "logs" / a.name
    opt.mkdir(parents=True, exist_ok=True)
    env = {"inp_text": lst, "inp_wav_dir": wavs, "exp_name": a.name, "opt_dir": opt, "i_part": 0, "all_parts": 1,
           "_CUDA_VISIBLE_DEVICES": 0, "is_half": True, "version": VERSION, "bert_pretrained_dir": BASE["bert"],
           "cnhubert_base_dir": BASE["hubert"], "sv_path": BASE["sv"], "pretrained_s2G": BASE["s2G"],
           "s2config_path": f"GPT_SoVITS/configs/s2{VERSION}.json"}
    run_step("prepare_datasets/1-get-text.py", env)
    (opt / "2-name2text-0.txt").replace(opt / "2-name2text.txt")
    run_step("prepare_datasets/2-get-hubert-wav32k.py", env)
    run_step("prepare_datasets/2-get-sv.py", env)
    run_step("prepare_datasets/3-get-semantic.py", env)
    part = opt / "6-name2semantic-0.tsv"
    body = part.read_text(encoding="utf-8").strip("\n")
    (opt / "6-name2semantic.tsv").write_text("item_name\tsemantic_audio\n" + body + "\n", encoding="utf-8")
    part.unlink()
    n_text = len((opt / "2-name2text.txt").read_text(encoding="utf-8").strip().splitlines())
    print(f"prepared {a.name}: {n_text} transcripts, {len(body.splitlines())} semantic sequences -> {opt}")


def train(a):
    import yaml

    opt = GSV / "logs" / a.name
    tmp = GSV / "TEMP"
    tmp.mkdir(exist_ok=True)
    for d in (f"SoVITS_weights_{VERSION}", f"GPT_weights_{VERSION}"):  # the WebUI makes them at start-up; the
        (GSV / d).mkdir(exist_ok=True)  # trainers only write into them (without them the e4 export failed)
    if a.sovits_epochs:
        cfg = json.loads((GSV / f"GPT_SoVITS/configs/s2{VERSION}.json").read_text(encoding="utf-8"))
        (opt / f"logs_s2_{VERSION}").mkdir(parents=True, exist_ok=True)
        cfg["train"].update({"batch_size": a.batch, "epochs": a.sovits_epochs, "text_low_lr_rate": 0.4,
                             "pretrained_s2G": BASE["s2G"], "pretrained_s2D": BASE["s2D"], "if_save_latest": True,
                             "if_save_every_weights": True, "save_every_epoch": a.save_every, "gpu_numbers": "0",
                             "grad_ckpt": False, "lora_rank": 32})
        cfg["model"]["version"] = VERSION
        cfg["data"]["exp_dir"] = cfg["s2_ckpt_dir"] = f"logs/{a.name}"
        cfg["save_weight_dir"] = f"SoVITS_weights_{VERSION}"
        cfg["name"], cfg["version"] = a.name, VERSION
        p = tmp / f"tmp_s2_{a.name}.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        print(f"=== SoVITS {a.sovits_epochs} epochs", flush=True)
        subprocess.run([sys.executable, "-s", "GPT_SoVITS/s2_train.py", "--config", str(p)], cwd=str(GSV), check=True,
                       env=gsv_env(_CUDA_VISIBLE_DEVICES=0, version=VERSION))
    if a.gpt_epochs:
        cfg = yaml.load((GSV / "GPT_SoVITS/configs/s1longer-v2.yaml").read_text(encoding="utf-8"), Loader=yaml.FullLoader)
        (opt / "logs_s1").mkdir(parents=True, exist_ok=True)
        cfg["train"].update({"batch_size": a.batch, "epochs": a.gpt_epochs, "save_every_n_epoch": a.save_every_gpt,
                             "if_save_every_weights": True, "if_save_latest": True, "if_dpo": False,
                             "half_weights_save_dir": f"GPT_weights_{VERSION}", "exp_name": a.name})
        cfg["pretrained_s1"] = BASE["s1"]
        cfg["train_semantic_path"] = f"logs/{a.name}/6-name2semantic.tsv"
        cfg["train_phoneme_path"] = f"logs/{a.name}/2-name2text.txt"
        cfg["output_dir"] = f"logs/{a.name}/logs_s1_{VERSION}"
        p = tmp / f"tmp_s1_{a.name}.yaml"
        p.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
        print(f"=== GPT {a.gpt_epochs} epochs", flush=True)
        subprocess.run([sys.executable, "-s", "GPT_SoVITS/s1_train.py", "--config_file", str(p)], cwd=str(GSV),
                       check=True, env=gsv_env(_CUDA_VISIBLE_DEVICES=0, hz="25hz", version=VERSION))
    use = DATA / a.name / "use.json"
    if not use.exists() and a.sovits_epochs and a.gpt_epochs:  # the first training sets the epochs speak uses;
        use.parent.mkdir(parents=True, exist_ok=True)       # a later run (more epochs to compare) does not
        use.write_text(json.dumps({"sovits_epoch": a.sovits_epochs, "gpt_epoch": a.gpt_epochs}), encoding="utf-8")
        print(f"speak will use SoVITS e{a.sovits_epochs} / GPT e{a.gpt_epochs} ({use})", flush=True)


def use_epochs(name):
    """The epochs speak takes by default: <gptsovits-data>/<name>/use.json, written by the first training (or by
    hand). Without it the highest epoch on disk was taken: myvoice_gsv1's SoVITS e16 - an over-trained comparison run,
    worse on the test set (PESQ 3.71 vs 3.98) - got into everything spoken after 03:00 on 2026-10-15."""
    p = DATA / name / "use.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def latest(folder, name, ext, epoch=None):
    pat = re.compile(re.escape(name) + r"[_-]e(\d+)")
    found = []
    for f in (GSV / folder).glob(f"{name}*{ext}"):
        m = pat.match(f.name)
        if m:
            found.append((int(m.group(1)), f))
    if not found:
        raise SystemExit(f"no {ext} weights for {name} in {folder}")
    if epoch:
        return next(f for e, f in found if e == epoch)
    return max(found)[1]


def speak(a):
    os.environ.update({k: v for k, v in gsv_env().items() if k in ("PATH", "PYTHONPATH")})
    a.list, a.out = str(Path(a.list).resolve()), str(Path(a.out).resolve())  # before the chdir
    os.chdir(GSV)
    sys.path[:0] = [str(GSV), str(GSV / "GPT_SoVITS")]
    import numpy as np
    import soundfile as sf
    from TTS_infer_pack.TTS import TTS, TTS_Config

    if a.base:
        gpt, sovits = BASE["s1"], BASE["s2G"]
    else:
        use = use_epochs(a.name)
        gpt = str(latest(f"GPT_weights_{VERSION}", a.name, ".ckpt", a.gpt_epoch or use.get("gpt_epoch")))
        sovits = str(latest(f"SoVITS_weights_{VERSION}", a.name, ".pth", a.sovits_epoch or use.get("sovits_epoch")))
    print(f"GPT {gpt}\nSoVITS {sovits}", flush=True)
    tts = TTS(TTS_Config({"custom": {"device": "cuda", "is_half": True, "version": VERSION, "t2s_weights_path": gpt,
                                      "vits_weights_path": sovits, "cnhuhbert_base_path": BASE["hubert"],
                                      "bert_base_path": BASE["bert"]}}))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    refs = DATA / "_refs"
    refs.mkdir(parents=True, exist_ok=True)
    for r in read_csv(a.list):
        speed = float(r["speed"]) if r.get("speed") else a.speed  # optional per-line column (tts_lines.py pace)
        src = Path(r["ref_path"])
        ref = refs / (hashlib.md5(str(src).encode("utf-8")).hexdigest()[:16] + ".wav")
        prompt = "" if a.no_prompt_text else r.get("ref_text", "")  # no text: GPT does not continue the prompt
        if not ref.exists():
            x, sr = sf.read(str(src), dtype="float32")
            if len(x) < 3.1 * sr:  # GPT-SoVITS refuses prompts outside 3-10 s: pad with silence
                x = np.concatenate([x, np.zeros((int(3.1 * sr) - len(x),) + x.shape[1:], dtype=x.dtype)])
            sf.write(str(ref), x, sr)
        x_len = sf.info(str(ref)).duration
        if x_len > 9.9:  # cut: the prompt text no longer matches the audio, so leave it out
            x, sr = sf.read(str(ref), dtype="float32")
            sf.write(str(ref), x[: int(9.9 * sr)], sr)
            prompt = ""
        for t in range(a.takes):  # --takes N: <key>_t<k>.wav, seeds seed..seed+N-1 (tts_pick.py chooses)
            name = r["key"] if a.takes == 1 else f"{r['key']}_t{t}"
            if a.skip_existing and (out / f"{name}.wav").exists():  # resume a long run
                continue
            req = {"text": r["text"], "text_lang": "ja", "ref_audio_path": str(ref), "prompt_text": prompt,
                   "prompt_lang": "ja", "top_k": a.top_k, "top_p": a.top_p, "temperature": a.temperature,
                   "text_split_method": "cut0", "batch_size": 1, "speed_factor": speed, "seed": a.seed + t,
                   "parallel_infer": False, "repetition_penalty": a.rep_penalty}
            chunks, sr = [], 32000
            for sr, audio in tts.run(req):
                chunks.append(audio)
            wav = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.int16)
            sf.write(out / f"{name}.wav", wav, sr, subtype="PCM_16")
            print(f"{name} {len(wav) / sr:.1f}s  {r['text']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prepare")
    s.add_argument("--name", required=True)
    s.add_argument("--exclude-flagged", action="store_true",
                   help="also leave out the clips clip_purity.py / transcript_check.py flagged (vl_paths.flagged_clips)")
    s.set_defaults(fn=prepare)
    s = sub.add_parser("train")
    s.add_argument("--name", required=True)
    s.add_argument("--sovits-epochs", type=int, default=8)
    s.add_argument("--gpt-epochs", type=int, default=15)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--save-every", type=int, default=4)
    s.add_argument("--save-every-gpt", type=int, default=5)
    s.set_defaults(fn=train)
    s = sub.add_parser("speak")
    s.add_argument("--name", default="")
    s.add_argument("--base", action="store_true", help="the pretrained models, not fine-tuned")
    s.add_argument("--gpt-epoch", type=int, default=None)
    s.add_argument("--sovits-epoch", type=int, default=None)
    s.add_argument("--list", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--top-k", type=int, default=15)
    s.add_argument("--top-p", type=float, default=1.0)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--rep-penalty", type=float, default=1.35)
    s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=1234)
    s.add_argument("--takes", type=int, default=1, help="takes per line, <key>_t<k>.wav (different seeds)")
    s.add_argument("--skip-existing", action="store_true", help="keep takes already written (resume a long run)")
    s.add_argument("--no-prompt-text", action="store_true",
                   help="leave the prompt's text out: GPT speaks from what it learned instead of continuing the "
                        "prompt's melody (the prompt still gives the timbre)")
    s.set_defaults(fn=speak)
    a = ap.parse_args()
    a.fn(a)
