"""Speak the game character's lines with the target voice's GPT-SoVITS model (tools env; VL_CHAR, VL_SET, VL_VOICE).

The character's original line is only a reference performance (the user, 2026-10-15: 第二个角色原句的音源仅作为参考):
  plan   per chosen line (--ids, or --all): emotion2vec and arousal cues of the original, each taken against her
         own speech lines (embedding centered on their mean, cues z-scored within them), then emotion_match's
         precise rule over the voice's clip bank -> the three best style references (3.2-9.5 s; clips flagged by
         clip_purity.py or transcript_check.py left out). A line longer than --unit-max (12 s; up to 64 s) is
         spoken phrase by phrase (the set's phrases.csv), each phrase taken against all her long lines' phrases.
         --reuse <work>: units with the same references there keep their takes. Writes <work>/plan.csv and
         speak_r0..r2.csv for gsv.py.
  (gsv.py speak per list, 3 takes each: with the reference's text -> <work>/takes_a_r<k>, with --no-prompt-text ->
   <work>/takes_np_r<k>)
  pick   --mode a      the 9 takes_a: timbre (ECAPA to her centroid) x2 + emotion2vec centered cosine to the first
                       reference + 0.5 same category - arousal-cue distance to it (loudness left out) - words (CER
                       over 0.1), as myvoice_gsv1_mr_t2 on the test set (the user's "A")
         --mode keyed  all 18: timbre x2 + SQUIM PESQ + the share of the original's key points the take shows x3
                       (keypoints.kept_share, DTW-aligned) + words, as myvoice_gsv1_keyed
         Writes <exp>/<group>/<id>.wav and pick.csv (with units: <group>/units/<unit>.wav, then each line
         assembled from its phrases at her gaps and relative levels); then superres.py and postprocess.py.
  pace   speaking rate like hers (the user, 2026-10-15: 做一下语速调整 - the trial's lines ran 0.64-1.26 x her
         length; the voice's model spoke 0.67-0.96 x her speaking time, the rest was silence): per line, the median over
         its takes in --work of their speaking time / hers (pauses left out) becomes gsv.py's speed for that line
         (0.75-1.35), written with plan.csv and the speak lists into --out-work; speak and pick there as before.
  retime the picked takes timed like her line: inner pauses set to the lengths of hers at the same place in the
         line (matched by relative position), lead and tail silences to hers; with --stretch atempo the take is
         first time-stretched to her speaking time. Writes <exp>/<group>_rt/ and retime.csv.
         Measured on two takes slowed to 0.75 / 0.8 (SQUIM PESQ, ECAPA to her centroid; take 3.76 / 0.733, 3.82 /
         0.769): Rubber Band (any setting) and librosa's phase vocoder 1.2-3.0 / 0.58-0.72 - phasey; ffmpeg atempo
         3.68 / 0.681, 3.80 / 0.735; gsv.py's own speed (new takes) 3.76-4.05 / 0.68-0.76, as at speed 1. Speaking
         a take again with the same seed at another speed is not possible: the same seed gave another take.
Measures are cached in <work>/_measures.json, so the two picks share them.

  set VL_CHAR=other & set VL_SET=all & set VL_VOICE=myvoice
  python scripts/tts_lines.py plan --ids L124,L130 --work data/other/tts/myvoice
  python scripts/tts_lines.py speak --work data/other/tts/myvoice
  python scripts/tts_lines.py pick --mode a --work data/other/tts/myvoice --group G_gsv_a
  python scripts/tts_lines.py retime --work data/other/tts/myvoice --groups G_gsv_a,G_gsv_keyed
"""
import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

import prosody
from arousal_features import features as arousal_raw
from emotion_match import AROUSAL, clip_path, precise_scores, sentence_type
from vl_common import EXP, HALLUCINATED, LINES, TARGET, normalize_ja, read_csv, read_mono, write_csv
from vl_paths import ROUTES, SET_DIR

SEEDS = {"a": 5000, "np": 5100}  # gsv.py --seed of takes_<kind>_r<k> is SEEDS[kind] + 10 k; take t adds t
PAUSE_MIN, SPEED_RANGE = 0.12, (0.75, 1.35)


def her_lines():
    """id -> (text, original wav) of the character's speech lines in this set."""
    return {r["id"]: (r["text"], LINES / f"{r['id']}.wav") for r in read_csv(ROUTES) if r["route"] == "speech"}


def flagged():
    """The voice's clips that clip_purity.py (another voice inside) or transcript_check.py (two ASRs disagree) flagged."""
    out = set()
    for name in ("purity.csv", "transcripts_check.csv"):
        p = TARGET / name
        if p.exists():
            out |= {r["id"] for r in read_csv(p) if r.get("flag")}
    return out


def units(ids, unit_max):
    """Speaking units: a line up to unit_max s whole (as in the trial), a longer one as its phrases
    (segment_lines.py: <set>/phrases/phrases.csv, cut at pauses and punctuation; Whisper's hallucinated
    「…」 removed from the text - L024's last phrase carried it)."""
    lines = her_lines()
    durs = {r["id"]: float(r["duration"]) for r in read_csv(ROUTES)}
    ph = {}
    for p in read_csv(SET_DIR / "phrases" / "phrases.csv"):
        ph.setdefault(p["line_id"], []).append(p)
    out = []
    for i in ids:
        text, wav = lines[i]
        if durs[i] <= unit_max or i not in ph:
            out.append({"id": i, "line": i, "kind": "line", "text": text, "perf": str(wav), "speech_start": "",
                        "speech_end": "", "line_dur": durs[i]})
            continue
        for p in sorted(ph[i], key=lambda p: int(p["idx"])):
            t = HALLUCINATED.sub("", p["text"]).strip()
            if normalize_ja(t):
                out.append({"id": p["id"], "line": i, "kind": "phrase", "text": t,
                            "perf": str(SET_DIR / "phrases" / f"{p['id']}.wav"), "speech_start": p["speech_start"],
                            "speech_end": p["speech_end"], "line_dur": durs[i]})
    return out


def plan(a):
    import shutil

    from tts_segments import embed_files

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    lines = her_lines()
    ids = list(lines) if a.all else [i for i in a.ids.split(",") if i]
    missing = [i for i in ids if i not in lines]
    if missing:
        raise SystemExit(f"not speech lines of this set: {missing}")
    num = lambda v: float(v) if v not in ("", None) else np.nan  # noqa: E731
    # her side: whole lines are taken against all her speech lines (as in the trial, so its lines get the same
    # references), phrases against all phrases of her long lines - the centering and z-scores do not depend on
    # which units are chosen
    corpus = units(list(lines), a.unit_max)
    whole = [{"id": i, "perf": str(p), "text": t} for i, (t, p) in lines.items()]
    phr = [u for u in corpus if u["kind"] == "phrase"]
    emo = embed_files([u["perf"] for u in whole + phr], work)
    stats = {}
    for kind, group in (("line", whole), ("phrase", phr)):
        if not group:
            continue
        raw = [arousal_raw(Path(u["perf"]), u["text"]) for u in group]
        H = np.array([[num(r[k]) if r else np.nan for k in AROUSAL] for r in raw])
        stats[kind] = {"mean": np.mean([emo[u["perf"]][0] for u in group], 0),
                       "aro": dict(zip([u["id"] for u in group], (H - np.nanmean(H, 0)) / (np.nanstd(H, 0) + 1e-9)))}
    # the voice's side: her clip bank, as tts_testset.py and tts_segments.py use it
    z = np.load(TARGET / "emo_bank.npz")
    bank_emo = {str(c): (z["feats"][i], z["scores"][i]) for i, c in enumerate(z["ids"])}
    ab = read_csv(TARGET / "arousal_bank.csv")
    A = np.array([[num(r[k]) for k in AROUSAL] for r in ab])
    b_aro = dict(zip([r["id"] for r in ab], (A - np.nanmean(A, 0)) / (np.nanstd(A, 0) + 1e-9)))
    bad = flagged()
    bank = [r for r in read_csv(TARGET / "train.csv") if r["id"] in bank_emo and r["id"] in b_aro and r["id"] not in bad
            and 3.2 <= float(r["duration"]) <= 9.5]
    B_e = np.stack([bank_emo[r["id"]][0] for r in bank])
    B_s = np.stack([bank_emo[r["id"]][1] for r in bank])
    B_a = np.stack([b_aro[r["id"]] for r in bank])
    B_t = np.array([sentence_type(r["text"]) for r in bank])
    reuse = {}
    if a.reuse:
        reuse = {r["id"]: r for r in read_csv(Path(a.reuse) / "plan.csv")}
    rows, lists = [], [[], [], []]
    for u in units(ids, a.unit_max):
        f, scores = emo[u["perf"]]
        st = stats[u["kind"]]
        total, _ = precise_scores(f, scores, st["aro"][u["id"]], sentence_type(u["text"]), B_e, B_s, B_a, B_t,
                                  st["mean"], z["feats"].mean(0))
        order = np.argsort(-total)[:3]
        rows.append({**u, "refs": " | ".join(bank[j]["id"] for j in order),
                     "ref_texts": " / ".join(bank[j]["text"] for j in order),
                     "scores": " ".join(f"{total[j]:.2f}" for j in order)})
        for k, j in enumerate(order):
            lists[k].append({"key": u["id"], "text": u["text"], "ref_path": str(clip_path(bank[j]["id"])),
                             "ref_text": bank[j]["text"]})
        # a line of an earlier run with the same references: its takes are copied, so its picks stay the ones
        # the user heard (the trial's 8 lines)
        old = reuse.get(u["id"])
        if old and old["refs"] == rows[-1]["refs"] and old["text"] == u["text"]:
            n = 0
            for t in Path(a.reuse).glob(f"takes_*_r*/{u['id']}_t*.wav"):
                dst = work / t.parent.name / t.name
                dst.parent.mkdir(exist_ok=True)
                if not dst.exists():
                    shutil.copy2(t, dst)
                    n += 1
            print(f"{u['id']}: {n} takes reused from {a.reuse}")
        elif old:
            print(f"{u['id']}: in {a.reuse} with other references or text - spoken again")
    write_csv(work / "plan.csv", rows)
    for k, lst in enumerate(lists):
        write_csv(work / f"speak_r{k}.csv", lst)
    n_lines = len({r["line"] for r in rows})
    print(f"{n_lines} lines as {len(rows)} units ({sum(r['kind'] == 'phrase' for r in rows)} phrases) -> "
          f"{work / 'plan.csv'}, speak_r0..2.csv ({len(bank)} clips in the bank, {len(bad)} flagged left out)")


def gsv(args):
    from tts_segments import GSV_PY

    subprocess.run([str(GSV_PY), str(Path(__file__).with_name("gsv.py")), "speak", *map(str, args)], check=True)


def speak(a):
    """3 takes per line with each of the three references: with the reference's text (takes_a_r<k>) and without
    (takes_np_r<k>, GPT speaks from what it learned instead of continuing the prompt's melody)."""
    work = Path(a.work)
    for kind, extra in (("a", []), ("np", ["--no-prompt-text"])):
        for k in range(3):
            gsv(["--name", a.model, "--takes", 3, "--seed", SEEDS[kind] + 10 * k, *extra, "--skip-existing",
                 "--list", work / f"speak_r{k}.csv", "--out", work / f"takes_{kind}_r{k}"])


def timing(x, sr, min_pause=PAUSE_MIN):
    """lead / end of the voiced span, its inner pauses [(start, end)] >= min_pause, the trailing silence and the
    speaking time without the pauses - seconds (silence: 30 dB under the loudest 10 ms, as tts_segments.pauses)."""
    from tts_segments import pauses

    h = int(0.01 * sr)
    db = 20 * np.log10(np.sqrt((x[: len(x) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9)
    on = np.where(db > db.max() - 30)[0]
    t0, t1 = on[0] * 0.01, (on[-1] + 1) * 0.01
    ps = [p for p in pauses(x, sr) if p[1] - p[0] >= min_pause]
    return {"lead": t0, "end": t1, "tail": len(x) / sr - t1, "pauses": ps,
            "speech": t1 - t0 - sum(q - p for p, q in ps if q - p >= PAUSE_MIN)}


def match_pauses(o, g):
    """(her pause, the take's pause) pairs at the same relative place in the voiced span (within 0.15 of it), in
    order; the take's candidates may be short (a pause she holds, the take only hints at)."""
    rel = lambda p, t: ((p[0] + p[1]) / 2 - t["lead"]) / max(t["end"] - t["lead"], 1e-6)  # noqa: E731
    pairs, last = [], -1
    for po in o["pauses"]:
        cands = [(abs(rel(po, o) - rel(pg, g)), j) for j, pg in enumerate(g["pauses"]) if j > last]
        if cands and min(cands)[0] < 0.15:
            j = min(cands)[1]
            pairs.append((po, g["pauses"][j]))
            last = j
    return pairs


def reshape(x, sr, o, g, pairs, edge=0.03, decay=0.08):
    """The take with each matched pause set to her length (its first and last 30 ms kept, silence between) and
    her lead and tail silences (the take's onset and 80 ms of its decay kept)."""
    e = int(edge * sr)
    pos = max(0, int(g["lead"] * sr) - e)
    out = [np.zeros(max(0, int((o["lead"] - edge) * sr)), dtype=np.float32)]
    for po, pg in pairs:
        a_, b_ = int(pg[0] * sr), int(pg[1] * sr)
        out.append(x[pos:a_ + e])
        out.append(np.zeros(max(0, int((po[1] - po[0]) * sr) - 2 * e), dtype=np.float32))
        pos = max(b_ - e, a_ + e)
    end = min(len(x), int((g["end"] + decay) * sr))
    out.append(x[pos:end])
    out.append(np.zeros(max(0, int((min(o["tail"], 0.6) - decay) * sr)), dtype=np.float32))
    return np.concatenate(out).astype(np.float32)


def pace(a):
    work, out = Path(a.work), Path(a.out_work)
    out.mkdir(parents=True, exist_ok=True)
    plan_rows = read_csv(work / "plan.csv")
    speed = {}
    for r in plan_rows:
        o = timing(*read_mono(r["perf"]))
        ratios = [timing(*read_mono(t))["speech"] / o["speech"] for t in takes_of(work, r["id"], ("a", "np"))]
        speed[r["id"]] = round(float(np.clip(np.median(ratios), *SPEED_RANGE)), 3)
        r["speed"] = speed[r["id"]]
        print(f"{r['id']} speaking time {np.median(ratios):.2f} x hers over {len(ratios)} takes -> speed {speed[r['id']]}")
    write_csv(out / "plan.csv", plan_rows)
    for k in range(3):
        write_csv(out / f"speak_r{k}.csv", [{**r, "speed": speed[r["key"]]} for r in read_csv(work / f"speak_r{k}.csv")])
    print(f"-> {out}: plan.csv, speak_r0..2.csv with a speed per line; now: speak --work {out}")


def complete(a):
    """A full set of the character's lines: speech lines from the GPT-SoVITS group (48 kHz, post-processed), every
    other line (refine, nonverbal: shouts, breaths - not text) from the current release <exp>/final_48k. Writes
    <exp>/<out>/<id>.wav + final.csv; final_48k itself (the live tool's and the wem export's source) is untouched."""
    import shutil

    from vl_common import load_lines
    from vl_paths import routes

    route, rel = routes(), EXP / "final_48k"
    rel_src = {r["id"]: r["source"] for r in read_csv(rel / "final.csv")}
    out = EXP / a.out
    out.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for ln in load_lines():
        r = route.get(ln["id"], "speech")
        src = EXP / a.group / f"{ln['id']}.wav" if r == "speech" else rel / f"{ln['id']}.wav"
        if not src.exists():
            missing.append(ln["id"])
            continue
        shutil.copy2(src, out / f"{ln['id']}.wav")
        # a line taken from the release keeps the release's source: "N_orig..." tells the live tool to leave the
        # original alone (9 short breaths), which "final_48k" would hide
        rows.append({"id": ln["id"], "src_wav": ln.get("src_wav", ""), "route": r,
                     "source": src.parent.name if r == "speech" else rel_src.get(ln["id"], "final_48k"),
                     "text": ln["ref_text"]})
    write_csv(out / "final.csv", rows)
    by = {}
    for r in rows:
        by[r["source"]] = by.get(r["source"], 0) + 1
    print(f"{a.out}: {len(rows)} lines {by}" + (f", MISSING {missing}" if missing else ""))


def stretch(src, dst, tempo):
    """ffmpeg atempo (overlap-add, pitch kept): tempo > 1 is faster. Rubber Band and phase vocoders smeared the voice's
    takes (PESQ 1.2-3.0), atempo kept them (3.7-3.8) - see the module docstring."""
    from vl_common import ROOT

    subprocess.run([str(ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"), "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(src), "-af", f"atempo={tempo:.4f}", str(dst)], check=True)


def retime(a):
    from tts_eval import cer
    from vl_common import transcribe, whisper_model

    work = Path(a.work)
    (work / "retimed").mkdir(exist_ok=True)
    asr = whisper_model()
    report = []
    for group in a.groups.split(","):
        out = EXP / f"{group}_rt"
        out.mkdir(parents=True, exist_ok=True)
        for p in read_csv(EXP / group / "pick.csv"):
            take = work / p["pick"]  # takes_a_r0/L124_t2.wav
            orig_x, orig_sr = read_mono(LINES / f"{p['id']}.wav")
            o, g0 = timing(orig_x, orig_sr), timing(*read_mono(take))
            tempo = float(np.clip(g0["speech"] / o["speech"], *SPEED_RANGE)) if a.stretch == "atempo" else 1.0
            mid = take
            if tempo != 1.0:
                mid = work / "retimed" / f"{group}__{p['id']}.wav"
                stretch(take, mid, tempo)
            x, sr = read_mono(mid)
            g = timing(x, sr, min_pause=0.04)
            pairs = match_pauses(o, g)
            y = reshape(x, sr, o, g, pairs)
            sf.write(out / f"{p['id']}.wav", y, sr, subtype="PCM_16")
            report.append({"group": group, "id": p["id"], "text": p["text"], "take": p["pick"], "tempo": round(tempo, 3),
                           "speech_ratio_before": round(g0["speech"] / o["speech"], 2),
                           "speech_ratio_after": round(g["speech"] / o["speech"], 2),
                           "length_ratio_before": round(len(read_mono(take)[0]) / read_mono(take)[1] / (len(orig_x) / orig_sr), 2),
                           "length_ratio_after": round(len(y) / sr / (len(orig_x) / orig_sr), 2),
                           "pauses_hers": len(o["pauses"]), "pauses_matched": len(pairs),
                           "cer": round(cer(p["text"], transcribe(asr, out / f"{p['id']}.wav")[0]), 2)})
            r = report[-1]
            print(f"{group} {p['id']} tempo {r['tempo']}  speech {r['speech_ratio_before']} -> {r['speech_ratio_after']}  "
                  f"length {r['length_ratio_before']} -> {r['length_ratio_after']}  pauses {len(pairs)}/{len(o['pauses'])}  "
                  f"cer {r['cer']}", flush=True)
    write_csv(work / "retime.csv", report)
    print(f"{len(report)} takes -> {EXP}/<group>_rt")


def takes_of(work, key, kinds):
    return sorted(t for kind in kinds for d in sorted(work.glob(f"takes_{kind}_r*")) for t in d.glob(f"{key}_t*.wav"))


def measure(work, plan_rows):
    """Per take: timbre, CER, PESQ, emotion2vec centered cosine and category against the first reference, arousal-cue
    distance to it, share of the original's key points kept. Cached by file stamp in <work>/_measures.json."""
    cache_p = work / "_measures.json"
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}
    stamp = lambda p: f"{p.stat().st_size}:{int(p.stat().st_mtime)}"  # noqa: E731
    first = {r["key"]: r for r in read_csv(work / "speak_r0.csv")}
    jobs = [(r, t) for r in plan_rows for t in takes_of(work, r["id"], ("a", "np"))]
    todo = [(r, t) for r, t in jobs if cache.get(str(t), {}).get("stamp") != stamp(t)]
    if todo:
        import torch
        import torchaudio
        from keypoints import kept_share, tracks
        from prep_voice import ecapa, embed
        from tts_eval import cer, emotions
        from vl_common import transcribe, whisper_model

        hold = set((TARGET / "tts" / "holdout_ids.txt").read_text(encoding="utf-8").splitlines())
        train = [r for r in read_csv(TARGET / "train.csv") if r["id"] not in hold]
        spk = ecapa()
        cen = np.mean([embed(spk, clip_path(r["id"])) for r in random.Random(1).sample(train, min(300, len(train)))], 0)
        cen /= np.linalg.norm(cen)
        asr = whisper_model()
        squim = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().eval()
        refs = {r["id"]: Path(first[r["id"]]["ref_path"]) for r in plan_rows}
        emo = emotions(sorted({str(t) for _, t in todo} | {str(p) for p in refs.values()}), work)
        mean_bank = np.load(TARGET / "emo_bank.npz")["feats"].mean(0)
        num = lambda v: float(v) if v not in ("", None) else np.nan  # noqa: E731
        sd = np.nanstd(np.array([[num(x[k]) for k in AROUSAL] for x in read_csv(TARGET / "arousal_bank.csv")]), 0) + 1e-9
        w = np.array([0.0 if k == "level" else v for k, v in AROUSAL.items()])
        ref_ar = {i: arousal_raw(p, first[i]["ref_text"]) for i, p in refs.items()}
        perf_tr = {r["id"]: tracks(r["perf"]) for r in plan_rows}
        for n, (r, t) in enumerate(todo, 1):
            x, sr = read_mono(t)
            with torch.inference_mode():
                q = float(squim(torch.from_numpy(np.ascontiguousarray(prosody._resample(x, sr), dtype=np.float32))[None])[1])
            fr, lr = emo[str(refs[r["id"]])]
            ft, lt = emo[str(t)]
            cr, ct = fr - mean_bank, ft - mean_bank
            at, ar = arousal_raw(t, r["text"]), ref_ar[r["id"]]
            dist = 9.0
            if at and ar:
                d = np.array([num(at[k]) - num(ar[k]) for k in AROUSAL]) / sd
                ok = np.isfinite(d) & (w > 0)
                dist = float(np.sqrt(np.sum(w[ok] * d[ok] ** 2) / max(w[ok].sum(), 1e-9)))
            cache[str(t)] = {"stamp": stamp(t), "timbre": float(embed(spk, t) @ cen), "pesq": q,
                             "cer": cer(r["text"], transcribe(asr, t)[0]),
                             "emo_cos": float(cr @ ct / (np.linalg.norm(cr) * np.linalg.norm(ct) + 1e-9)),
                             "emo_same": float(lt == lr), "emo_cat": lt, "ref_cat": lr, "arousal": dist,
                             "keys": kept_share(r["perf"], t, perf_tr[r["id"]])}
            if n % 20 == 0:
                print(f"  measured {n}/{len(todo)}", flush=True)
                cache_p.write_text(json.dumps(cache), encoding="utf-8")
        cache_p.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def pick(a):
    import shutil

    from tts_pick import z

    work = Path(a.work)
    plan_rows = read_csv(work / "plan.csv")
    m = measure(work, plan_rows)
    out = EXP / a.group
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for r in plan_rows:
        takes = takes_of(work, r["id"], ("a",) if a.mode == "a" else ("a", "np"))
        if not takes:
            print(f"{r['id']}: no takes")
            continue
        v = lambda k: np.array([m[str(t)][k] for t in takes])  # noqa: E731
        words = -30 * np.maximum(0, v("cer") - 0.1)
        if a.mode == "a":
            score = 2 * z(v("timbre")) + z(v("emo_cos")) + 0.5 * v("emo_same") - z(v("arousal")) + words
        else:
            score = 2 * z(v("timbre")) + z(v("pesq")) + 3 * z(v("keys")) + words
        k = int(np.argmax(score))
        dst = out / "units" / f"{r['id']}.wav" if r.get("kind") else out / f"{r['id']}.wav"
        dst.parent.mkdir(exist_ok=True)
        shutil.copy2(takes[k], dst)
        mk = m[str(takes[k])]
        report.append({"id": r["id"], "line": r.get("line", r["id"]), "text": r["text"],
                       "pick": f"{takes[k].parent.name}/{takes[k].name}",
                       "timbre": round(mk["timbre"], 3), "cer": round(mk["cer"], 2), "pesq": round(mk["pesq"], 2),
                       "keys": round(mk["keys"], 2), "emo": f"{mk['emo_cat']} / {mk['ref_cat']}",
                       "emo_cos": round(mk["emo_cos"], 2), "arousal": round(mk["arousal"], 2)})
        print(f"{r['id']} -> {report[-1]['pick']}  timbre {mk['timbre']:.3f} keys {mk['keys']:.2f} cer {mk['cer']:.2f}")
    write_csv(out / "pick.csv", report)
    if any(r.get("kind") for r in plan_rows):
        assemble(plan_rows, out)
    print(f"{len(report)} units -> {out}")


def assemble(plan_rows, out, sr=32000, fade=0.005):
    """<out>/units/ -> <out>/<line>.wav: a whole-line unit as it is; a long line's phrases, silence trimmed, at her
    gaps between them (her speech_start / speech_end of each phrase), her lead and tail, each at her level
    relative to her loudest phrase (active RMS: 10 ms frames within 30 dB of the loudest). The speech itself is
    not stretched (the user, 2026-10-15: 用没有调速的 a 和关键点)."""
    import shutil

    import librosa

    def active_rms(x):
        h = int(0.01 * sr)
        if len(x) < h:
            return 1e-9
        f = x[: len(x) // h * h].reshape(-1, h)
        e = np.sqrt((f ** 2).mean(1)) + 1e-9
        return float(np.sqrt(np.mean(e[e > e.max() * 10 ** (-30 / 20)] ** 2)))

    by = {}
    for r in plan_rows:
        by.setdefault(r["line"], []).append(r)
    for line, us in by.items():
        src = [out / "units" / f"{u['id']}.wav" for u in us]
        if not all(p.exists() for p in src):
            print(f"{line}: units missing, not assembled")
            continue
        if len(us) == 1 and us[0]["kind"] == "line":
            shutil.copy2(src[0], out / f"{line}.wav")
            continue
        hers = [active_rms(read_mono(u["perf"], sr)[0]) for u in us]
        n_f = int(fade * sr)
        parts = [np.zeros(int(float(us[0]["speech_start"]) * sr), np.float32)]
        for i, (u, p) in enumerate(zip(us, src)):
            x = read_mono(p, sr)[0]
            _, (b0, b1) = librosa.effects.trim(x, top_db=40)
            x = x[b0:b1].copy()
            x *= 0.1 * hers[i] / max(hers) / (active_rms(x) + 1e-9)
            x[:n_f] *= np.linspace(0, 1, min(n_f, len(x)))
            x[-n_f:] *= np.linspace(1, 0, min(n_f, len(x)))
            parts.append(x.astype(np.float32))
            nxt = float(us[i + 1]["speech_start"]) if i + 1 < len(us) else float(u["line_dur"])
            parts.append(np.zeros(int(max(0.08, nxt - float(u["speech_end"])) * sr), np.float32))
        y = np.concatenate(parts)
        if np.abs(y).max() > 0.95:
            y *= 0.95 / np.abs(y).max()
        sf.write(out / f"{line}.wav", y, sr, subtype="PCM_16")
    print(f"{len(by)} lines assembled -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plan")
    s.add_argument("--ids", default="", help="comma-separated line ids (routes.csv, route speech)")
    s.add_argument("--all", action="store_true", help="every speech line of the set")
    s.add_argument("--unit-max", type=float, default=12.0, help="lines longer than this are spoken phrase by phrase")
    s.add_argument("--reuse", default="", help="an earlier work dir: lines with the same references keep its takes")
    s.add_argument("--work", required=True)
    s.set_defaults(fn=plan)
    s = sub.add_parser("pick")
    s.add_argument("--mode", choices=["a", "keyed"], required=True)
    s.add_argument("--work", required=True)
    s.add_argument("--group", required=True, help="output group under the experiment dir, e.g. G_gsv_a")
    s.set_defaults(fn=pick)
    s = sub.add_parser("speak")
    s.add_argument("--work", required=True)
    s.add_argument("--model", default="myvoice_gsv1")
    s.set_defaults(fn=speak)
    s = sub.add_parser("retime")
    s.add_argument("--work", required=True)
    s.add_argument("--groups", required=True, help="picked groups, e.g. G_gsv_a,G_gsv_keyed -> <group>_rt")
    s.add_argument("--stretch", choices=["none", "atempo"], default="none",
                   help="none: pauses and silences only (the rate set at generation by pace); atempo: also stretch")
    s.set_defaults(fn=retime)
    s = sub.add_parser("complete")
    s.add_argument("--group", required=True, help="the GPT-SoVITS group for the speech lines, e.g. G_full_a_48k")
    s.add_argument("--out", required=True, help="the full set's group, e.g. final_gsv_a_48k")
    s.set_defaults(fn=complete)
    s = sub.add_parser("pace")
    s.add_argument("--work", required=True, help="a work dir with takes (its plan.csv and speak lists are copied)")
    s.add_argument("--out-work", required=True)
    s.set_defaults(fn=pace)
    a = ap.parse_args()
    a.fn(a)
