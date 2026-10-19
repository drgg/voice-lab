"""Refine one line, step 3: score every take per phrase and build a pick-and-play page (tools env).

Takes come from several exp/ folders (IndexTTS sets, their RVC polish, voice conversion of the
original phrases); files are <pid>.wav or <pid>_c<k>.wav. Each take is scored like
select_phrases.py, trimmed and levelled to the original phrase, and copied to
exp/refine_<line>/cands/. The page (eval/refine_<line>.html) lists the best --top takes per phrase;
the listener picks one per phrase and plays the combination on the original timing in the browser,
then copies the pick code. The auto pick (best score per phrase) is rendered as <line>_auto.wav.

  envs/tools/Scripts/python scripts/make_refine_page.py --line L14 --sets "L14_tts_a10=IndexTTS 情感1.0,..."
  envs/tools/Scripts/python scripts/make_refine_page.py --line L14 --pick "1=1-03,2=2-07,3=3-01"
      -> exp/refined/<line>.wav (run_refine.ps1 -Pick then post-processes and super-resolves it)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from select_phrases import SR, place, register_offset, rms, score_take, trim
from vl_common import EVAL, EXP, LINES, REFINE, read_csv, read_mono, whisper_model
from vl_paths import page_path, voice_cfg


def phrases_of(line):
    return read_csv(REFINE / line / "phrases.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", required=True)
    ap.add_argument("--sets", default="", help="folder=label,... under exp/")
    ap.add_argument("--top", type=int, default=8, help="takes shown per phrase")
    ap.add_argument("--pick", default=None, help="assemble the listener's choice, e.g. 1=1-03,2=2-07,3=3-01")
    a = ap.parse_args()

    work = EXP / f"refine_{a.line}"
    phrases = phrases_of(a.line)
    line_wav = read_mono(LINES / f"{a.line}.wav", SR)[0]
    if a.pick:
        keys = dict(kv.split("=") for kv in a.pick.split(","))
        takes = {p["id"]: work / "cands" / f"{keys[p['idx']]}.wav" for p in phrases}
        final = place(phrases, takes, line_wav)
        sf.write(work / f"{a.line}_final.wav", final, SR, subtype="PCM_16")
        (EXP / "refined").mkdir(parents=True, exist_ok=True)  # collected for postprocess / AudioSR / finalize
        sf.write(EXP / "refined" / f"{a.line}.wav", final, SR, subtype="PCM_16")
        (work / "pick.txt").write_text(a.pick, encoding="utf-8")
        print(f"assembled {keys} -> {EXP / 'refined' / f'{a.line}.wav'}")
        raise SystemExit

    (work / "cands").mkdir(parents=True, exist_ok=True)
    for old in (work / "cands").glob("*.wav"):
        old.unlink()
    sets = [s.split("=", 1) for s in a.sets.split(",") if s]
    offset, asr = register_offset(), whisper_model()
    line_rms = rms(trim(line_wav, SR))
    data = {"line": a.line, "text": "".join(p["text"] for p in phrases), "orig": page_path(LINES / f"{a.line}.wav"),
            "auto": page_path(work / f"{a.line}_auto.wav"), "phrases": []}
    auto = {}
    for p in phrases:
        o_wav, o_sr = read_mono(REFINE / a.line / f"{p['id']}.wav")
        o_sp = trim(o_wav, o_sr)
        o_seg = line_wav[int(float(p["speech_start"]) * SR):int(float(p["speech_end"]) * SR)]
        takes = []
        for folder, label in sets:
            for f in sorted((EXP / folder).glob(f"{p['id']}*.wav")):
                if f.stem != p["id"] and not f.stem.startswith(p["id"] + "_c"):
                    continue
                m, hyp = score_take(asr, f, p["text"], o_sp, o_sr, offset)
                take_no = f" · 第 {int(f.stem.rsplit('_c', 1)[1]) + 1} 版" if "_c" in f.stem[len(p["id"]):] else ""
                takes.append({"file": f, "label": label + take_no, "asr": hyp, **{k: round(v, 3) for k, v in m.items()}})
        takes.sort(key=lambda t: -t["score"])
        shown = takes[:a.top]
        for folder, label in sets:  # keep at least one take of every kind on the page
            if not any(t["label"].startswith(label) for t in shown):
                shown += [t for t in takes if t["label"].startswith(label)][:1]
        items = []
        for k, t in enumerate(shown, 1):
            key = f"{p['idx']}-{k:02d}"
            seg = trim(read_mono(t["file"], SR)[0], SR)
            seg = seg * (rms(o_seg) / line_rms) / (rms(seg) / 0.1)
            sf.write(work / "cands" / f"{key}.wav", np.clip(seg, -0.99, 0.99), SR, subtype="PCM_16")
            items.append({"key": key, "label": t["label"], "src": page_path(work / "cands" / f"{key}.wav"),
                          "score": t["score"], "corr": t["corr"], "cer": t["cer"], "asr": t["asr"]})
            print(f"{p['id']} {key} {t['score']:+.2f} corr={t['corr']:.2f} cer={t['cer']:.2f} {t['label']} [{t['asr']}]")
        auto[p["id"]] = work / "cands" / f"{items[0]['key']}.wav"
        data["phrases"].append({"idx": p["idx"], "pid": p["id"], "text": p["text"], "start": float(p["speech_start"]),
                                "orig": page_path(REFINE / a.line / f"{p['id']}.wav"),
                                "auto": items[0]["key"], "cands": items})
    sf.write(work / f"{a.line}_auto.wav", place(phrases, auto, line_wav), SR, subtype="PCM_16")
    tpl = (Path(__file__).parent / "templates" / "refine.html").read_text(encoding="utf-8")
    EVAL.mkdir(parents=True, exist_ok=True)
    out = EVAL / f"refine_{a.line}.html"
    out.write_text(tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__VOICE__", voice_cfg()["label"]),
                   encoding="utf-8")
    print(f"page -> {out}")
