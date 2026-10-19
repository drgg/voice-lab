"""Speaker embeddings from several models, for choosing the one that tells the target voice from the show's other
characters best (seedvc env: funasr / modelscope / transformers). The user, 2026-10-18: 有没有办法用更多的时间在原剧中
更加精细的提取声线 - step 1 of that plan compares models on the clips the user labelled by ear.

Per wav: one embedding of the whole clip and one per 1.5 s window (0.5 s hop) over it, L2-normalised.
  campplus    CAM++ (3D-Speaker), iic/speech_campplus_sv_zh-cn_16k-common, ~200k Mandarin speakers (through FunASR)
  wavlm       microsoft/wavlm-base-plus-sv (x-vector head on WavLM Base+)
  eres2netv2  ERes2NetV2 (3D-Speaker, w24s4ep4): the copy GPT-SoVITS v2ProPlus ships for its speaker conditioning, run
              from its code in the gptsovits env (GSV_ROOT); `forward` = the 192-d verification embedding
              (GPT-SoVITS itself uses forward3, the pooled feature map). Not in FunASR 1.1.5's registry.
  redimnet2   ReDimNet2 B6, large-margin fine-tuned on VoxBlink2 + VoxCeleb2 + CN-Celeb (2026, MIT), torch.hub from
              github.com/PalabraAI/redimnet2 (code and v1.0.0 release weights, TORCH_HOME = cache/torch)
Models are fetched once from ModelScope / Hugging Face / GitHub into cache/ (MODELSCOPE_CACHE, HF_HOME, TORCH_HOME).

  envs/seedvc/Scripts/python scripts/speaker_embed.py --model campplus --list files.txt --out emb_campplus.npz
  E:/voice-lab-cache/envs/gptsovits/Scripts/python scripts/speaker_embed.py --model eres2netv2 ...
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT / "cache" / "modelscope"))
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(ROOT / "cache" / "hf" / "hub"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "cache" / "torch"))
GSV_ROOT = Path(os.environ.get("GSV_ROOT", "E:/voice-lab-cache/GPT-SoVITS"))

import numpy as np
import soundfile as sf
import torch

SR = 16000
WIN, HOP = 1.5, 0.5
CAMPPLUS = "iic/speech_campplus_sv_zh-cn_16k-common"


def load16(path, start=None, end=None):
    """The file (or its start..end seconds) as 16 kHz mono."""
    import torchaudio.functional as AF
    if start is None:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    else:
        sr = sf.info(str(path)).samplerate
        x, _ = sf.read(str(path), start=int(start * sr), stop=int(end * sr), dtype="float32", always_2d=True)
    x = torch.from_numpy(x.mean(axis=1))
    if sr != SR:
        x = AF.resample(x, sr, SR)
    return x.numpy()


class Embedder:
    def __init__(self, name):
        self.name = name
        if name == "campplus":
            from funasr import AutoModel
            self.net = AutoModel(model=CAMPPLUS, device="cuda", disable_update=True)
        elif name == "wavlm":
            from transformers import AutoFeatureExtractor, WavLMForXVector
            self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
            self.net = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").cuda().eval()
        elif name == "eres2netv2":
            sys.path.insert(0, str(GSV_ROOT / "GPT_SoVITS" / "eres2net"))
            from ERes2NetV2 import ERes2NetV2
            import kaldi
            self.kaldi = kaldi
            st = torch.load(GSV_ROOT / "GPT_SoVITS" / "pretrained_models" / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt",
                            map_location="cpu", weights_only=False)
            self.net = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
            self.net.load_state_dict(st)
            self.net = self.net.cuda().eval()
        elif name == "redimnet2":
            self.net = torch.hub.load("PalabraAI/redimnet2", "redimnet2", model_name="b6", train_type="lm",
                                      dataset="vb2+vox2+cnc2_v0", pretrained=True, trust_repo=True).cuda().eval()
        else:
            raise SystemExit(f"unknown model {name}")

    @torch.no_grad()
    def __call__(self, wavs):
        """wavs: list of 16 kHz float arrays -> (n, d) normalised embeddings."""
        out = []
        for w in wavs:
            if len(w) < int(0.3 * SR):
                w = np.pad(w, (0, int(0.3 * SR) - len(w)))
            if self.name == "wavlm":
                inp = self.fe(w, sampling_rate=SR, return_tensors="pt").to("cuda")
                e = self.net(**inp).embeddings[0]
            elif self.name == "eres2netv2":
                feat = self.kaldi.fbank(torch.from_numpy(w)[None], num_mel_bins=80, sample_frequency=SR, dither=0)
                e = self.net(feat[None].cuda())[0]
            elif self.name == "redimnet2":
                e = self.net(torch.from_numpy(w)[None].cuda())[0]
            else:
                e = self.net.generate(input=w.astype(np.float32), disable_pbar=True)[0]["spk_embedding"][0]
            e = e.float().cpu().numpy()
            out.append(e / (np.linalg.norm(e) + 1e-9))
        return np.stack(out)


def windows(x):
    n, h = int(WIN * SR), int(HOP * SR)
    if len(x) <= n:
        return [x]
    return [x[i:i + n] for i in range(0, len(x) - n + 1, h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["campplus", "wavlm", "eres2netv2", "redimnet2"])
    ap.add_argument("--list", required=True, help="text file: one wav path per line, or path<TAB>start<TAB>end (s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-windows", action="store_true", help="whole-clip embeddings only (half the time)")
    a = ap.parse_args()
    paths = [p.rstrip("\r\n") for p in open(a.list, encoding="utf-8") if p.strip()]
    emb = Embedder(a.model)
    E, W = [], []
    for k, p in enumerate(paths):
        f = p.split("\t")
        x = load16(f[0]) if len(f) == 1 else load16(f[0], float(f[1]), float(f[2]))
        E.append(emb([x])[0])
        if not a.no_windows:
            W.append(emb(windows(x)))
        if k % 500 == 0:
            print(f"{a.model}: {k}/{len(paths)}", flush=True)
    np.savez(a.out, E=np.stack(E), W=np.array(W, dtype=object), paths=np.array(paths))
    print(f"{a.model}: {len(paths)} clips, dim {E[0].shape[0]} -> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
