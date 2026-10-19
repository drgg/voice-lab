"""Speak a line segment by segment, each with its own style reference (tools env; VL_VOICE; GPT-SoVITS via gsv.py).

One style reference per line gave the whole line one mood: the user heard 「…」
(the voice's test line T15) start a little angry and end in a question in her performance, but come out angry throughout,
and T51 join a long sentence and a short question into one tone (2026-10-15). Here a reference performance of the
line - the game character's original, or for the test set her own clip (then an upper bound) - only guides:
  plan   cut the performance at its pauses (>= 0.2 s at an accent-phrase boundary - pyopenjtalk -, or >= 0.06 s at
         a comma / ? / ! in the words), merge pieces under 0.6 s; words from Whisper's word timestamps. A piece under
         1 s is spoken with its neighbour as a carrier and carved out at the pause between them (pick). Per segment: emotion2vec + arousal cues of that piece,
         emotion_match's precise rule over her clip bank -> two style references (3.2-9.5 s, not from the line's
         own scene). Writes <work>/segments.csv and speak lists for gsv.py.
  pick   per segment, among its takes: timbre (ECAPA to her centroid) x2, SQUIM PESQ, words (CER of the segment),
         the shape of the pitch at its start and end against the performance's (first 0.4 s / last 0.6 s of voicing,
         relative to each one's median, 12 points; only the shape: registers differ), her ending habit for the last
         segment (tts_pick.ending_score), emotion2vec against the performance piece (centered cosine, + same
         category x0.5) and its arousal cues (distance, loudness left out). The first run scored without the
         emotion terms and lost the emotion match whole-line picking had (2026-10-15). Measures are cached in
         <work>/_measures.json. Then stitch the picks at the performance's pauses and relative levels.

  python scripts/tts_segments.py plan --set test --work data/<voice>/tts/seg
  ($VL_CACHE/envs/gptsovits ... gsv.py speak --list <work>/speak_r0.csv --takes 3 --out <work>/takes_r0, and _r1)
  python scripts/tts_segments.py pick --set test --work data/<voice>/tts/seg --out data/<voice>/tts/gen/<name>
"""
import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import prosody
from arousal_features import features as arousal_raw
from emotion_match import AROUSAL, clip_path, precise_scores, sentence_type
from vl_common import ROOT, TARGET, read_csv, read_mono, whisper_model, write_csv

PAUSE, PAUSE_PUNCT, MIN_SEG, SHORT = 0.15, 0.06, 0.6, 1.0
PARTICLE = re.compile(r"(よ|ね|わ|の|か|さ|な|ぞ|ぜ|から|けど|って|て|で|が|は|に|を|も|[、,，。?？!！…])$")
GSV_PY = Path(r"E:\voice-lab-cache\envs\gptsovits\Scripts\python.exe")


def lines_of(which):
    """[(key, text, performance wav, scene ids to avoid)] - the held-out test set ('test')."""
    if which == "test":
        hold = set((TARGET / "tts" / "holdout_ids.txt").read_text(encoding="utf-8").splitlines())
        return [(r["key"], r["text"], clip_path(r["id"]), hold) for r in read_csv(TARGET / "tts" / "testset.csv")]
    raise SystemExit(f"unknown set {which}")


def pauses(wav, sr):
    """[(start, end)] of the silences inside the voiced span (10 ms frames, 30 dB under the loudest)."""
    h = int(0.01 * sr)
    fr = wav[: len(wav) // h * h].reshape(-1, h)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
    on = db > db.max() - 30
    idx = np.where(on)[0]
    if not len(idx):
        return []
    out, start = [], None
    for i in range(idx[0], idx[-1] + 1):
        if not on[i] and start is None:
            start = i
        elif on[i] and start is not None:
            out.append((start * 0.01, i * 0.01))
            start = None
    return out


def phrase_starts(texts):
    """Character offsets where an accent phrase begins (pyopenjtalk's chain flags, GPT-SoVITS env), per text; None
    where the analysis does not cover the text one character for one. Whisper's words are sub-word pieces: cutting
    at any of their boundaries split 「…」 (T13), 「…」 (T38), 「…」 (T59)."""
    code = ("import json, sys, pyopenjtalk\n"
            "out = []\n"
            "for t in json.loads(sys.stdin.read()):\n"
            "    f = pyopenjtalk.run_frontend(t) if t else []\n"
            "    pos, st = 0, []\n"
            "    for x in f:\n"
            "        if x['chain_flag'] != 1 and pos:\n"
            "            st.append(pos)\n"
            "        pos += len(x['string'])\n"
            "    out.append(st if pos == len(t) else None)\n"
            "sys.stdout.write(json.dumps(out))\n")
    r = subprocess.run([str(GSV_PY), "-c", code], input=json.dumps(texts), capture_output=True, text=True,
                       encoding="utf-8", check=True)
    return [set(x) if x is not None else None for x in json.loads(r.stdout)]


def segment(words, wav, sr, starts=None):
    """Cut points: after a word ending in punctuation, the longest pause (>= PAUSE_PUNCT) starting near its end;
    any pause >= 0.2 s at an accent-phrase boundary (starts: character offsets, phrase_starts) near it, after a
    particle if there is one. Whisper's word times are loose (T15: 「…」 stamped 1.70-2.42 s across
    a 0.44 s pause at 1.79 s), so pauses are matched by where they start. Then pieces under MIN_SEG are merged into
    a neighbour."""
    ps = pauses(wav, sr)
    offs = np.cumsum([0] + [len(w["word"].strip()) for w in words])
    ok = lambda j: starts is None or int(offs[j]) in starts  # noqa: E731
    cuts = {}  # first word index after the cut -> (pause start, pause end)
    for i, w in enumerate(words[:-1]):
        if re.search(r"[、,，。?？!！…]$", w["word"].strip()):
            near = [p for p in ps if w["end"] - 0.15 <= p[0] <= w["end"] + 0.4 and p[1] - p[0] >= PAUSE_PUNCT]
            if near:
                cuts[i + 1] = max(near, key=lambda p: p[1] - p[0])
    for p0, p1 in ps:
        if p1 - p0 >= 0.2 and len(words) > 1 and not any(abs(c[0] - p0) < 0.05 for c in cuts.values()):
            # phrase boundaries near the pause; after a particle if there is one (T03's pause was stamped inside
            # 「シ」: the nearest boundary cut 「…」, the particle rule cuts after 「…」)
            near = [j for j in range(1, len(words)) if words[j]["start"] >= p0 - 0.5 and words[j - 1]["end"] <= p1 + 0.1
                    and ok(j)]
            after = [j for j in near if PARTICLE.search(words[j - 1]["word"].strip())]
            if not (after or near):
                continue
            j = min(after or near, key=lambda j: abs(words[j]["start"] - (p0 + p1) / 2))
            if j not in cuts or cuts[j][1] - cuts[j][0] < p1 - p0:
                cuts[j] = (p0, p1)
    cuts = [(p[0], p[1], j) for j, p in sorted(cuts.items())]
    segs, first, t0 = [], 0, 0.0
    for p0, p1, wi in cuts + [(len(wav) / sr, len(wav) / sr, len(words))]:
        segs.append({"w0": first, "w1": wi, "start": t0, "end": p0, "next": p1})
        first, t0 = wi, p1
    merged = []
    for s in segs:  # a piece too short to carry a mood joins its neighbour
        if merged and (s["end"] - s["start"] < MIN_SEG or merged[-1]["end"] - merged[-1]["start"] < MIN_SEG):
            merged[-1].update({"w1": s["w1"], "end": s["end"], "next": s["next"]})
        else:
            merged.append(dict(s))
    for s in merged:
        s["text"] = "".join(w["word"] for w in words[s["w0"]:s["w1"]]).strip()
    return [s for s in merged if s["text"]]


def shape(x, sr, part, n=12):
    st, _ = prosody.semitones(x, sr)
    v = np.where(np.isfinite(st))[0]
    if len(v) < 8:
        return None
    seg = st[v[:40]] if part == "start" else st[v[-60:]]
    seg = seg - np.nanmedian(st)
    return np.interp(np.linspace(0, len(seg) - 1, n), np.arange(len(seg)), seg)


def plan(a):
    work = Path(a.work)
    (work / "perf").mkdir(parents=True, exist_ok=True)
    asr = whisper_model()
    prev = {f"{r['key']}_s{r['seg']}": r for r in read_csv(work / "segments.csv")} if (work / "segments.csv").exists() else {}
    lines = []
    for key, text, perf, _ in lines_of(a.set):
        segs, _ = asr.transcribe(str(perf), language="ja", word_timestamps=True, beam_size=5, condition_on_previous_text=False)
        lines.append((key, text, perf, [{"word": w.word, "start": w.start, "end": w.end} for s in segs for w in (s.words or [])]))
    starts = phrase_starts(["".join(w["word"].strip() for w in words) for *_, words in lines])
    rows = []
    for (key, text, perf, words), st in zip(lines, starts):
        wav, sr = read_mono(perf)
        parts = segment(words, wav, sr, st) if words else [{"text": text, "start": 0.0, "end": len(wav) / sr, "next": len(wav) / sr}]
        if len(parts) == 1:
            parts[0]["text"] = text  # one piece: the line's own text (Whisper's word split may drop punctuation)
        for i, s in enumerate(parts):
            p = work / "perf" / f"{key}_s{i}.wav"
            sf.write(p, wav[int(s["start"] * sr):int(s["end"] * sr)], sr, subtype="PCM_16")
            # a piece under SHORT s spoken alone came out garbled (T51's 「…」: every take misread, ECAPA 0.1-0.3):
            # it is spoken with its neighbour as a carrier and carved out of the take at the pause between them
            side, say = "", s["text"]
            if len(parts) > 1 and s["end"] - s["start"] < SHORT:
                side = "before" if i else "after"
                left, right = (parts[i - 1]["text"], s["text"]) if i else (s["text"], parts[i + 1]["text"])
                say = left + ("" if re.search(r"[、,，。?？!！…]$", left) else "、") + right
            rows.append({"key": key, "seg": i, "n_seg": len(parts), "text": s["text"], "start": round(s["start"], 3),
                         "end": round(s["end"], 3), "next": round(s["next"], 3), "perf": str(p), "carrier": side,
                         "say": say})
    write_csv(work / "segments.csv", rows)
    print(f"{len(set(r['key'] for r in rows))} lines -> {len(rows)} segments "
          f"({sum(1 for r in rows if r['n_seg'] > 1)} in lines of 2+)")
    # style references per segment: the precise rule over her bank, as tts_testset.py does per line
    emo = embed_files([r["perf"] for r in rows], work)
    z = np.load(TARGET / "emo_bank.npz")
    bank_emo = {str(c): (z["feats"][i], z["scores"][i]) for i, c in enumerate(z["ids"])}
    num = lambda v: float(v) if v not in ("", None) else np.nan  # noqa: E731
    ab = read_csv(TARGET / "arousal_bank.csv")
    A = np.array([[num(r[k]) for k in AROUSAL] for r in ab])
    mu, sd = np.nanmean(A, 0), np.nanstd(A, 0) + 1e-9
    aro = {r["id"]: (v - mu) / sd for r, v in zip(ab, A)}
    hold = lines_of(a.set)[0][3]
    train = [r for r in read_csv(TARGET / "train.csv") if r["id"] not in hold and r["id"] in bank_emo and r["id"] in aro
             and 3.2 <= float(r["duration"]) <= 9.5]
    B_e = np.stack([bank_emo[r["id"]][0] for r in train])
    B_s = np.stack([bank_emo[r["id"]][1] for r in train])
    B_a = np.stack([aro[r["id"]] for r in train])
    B_t = np.array([sentence_type(r["text"]) for r in train])
    mean_bank = z["feats"].mean(0)
    lists = [[], []]
    for r in rows:
        f, scores = emo[r["perf"]]
        raw = arousal_raw(Path(r["perf"]), r["text"])
        a_seg = (np.array([num(raw[k]) for k in AROUSAL]) - mu) / sd if raw else np.full(len(AROUSAL), np.nan)
        total, _ = precise_scores(f, scores, a_seg, sentence_type(r["text"]), B_e, B_s, B_a, B_t, mean_bank, mean_bank)
        order = np.argsort(-total)[:2]
        r["refs"] = " | ".join(train[j]["id"] for j in order)
        for k, j in enumerate(order):
            lists[k].append({"key": f"{r['key']}_s{r['seg']}", "text": r["say"], "ref_path": str(clip_path(train[j]["id"])),
                             "ref_text": train[j]["text"]})
    write_csv(work / "segments.csv", rows)
    # segments whose words, carrier or references changed since the last plan: only these need new takes
    def same(r, o):
        return bool(o) and r["text"] == o["text"] and r["say"] == (o.get("say") or o["text"]) and r["refs"] == o.get("refs")

    redo = {f"{r['key']}_s{r['seg']}" for r in rows if not same(r, prev.get(f"{r['key']}_s{r['seg']}"))}
    for k, lst in enumerate(lists):
        write_csv(work / f"speak_r{k}.csv", lst)
        write_csv(work / f"speak_r{k}_redo.csv", [x for x in lst if x["key"] in redo])
    print(f"speak lists: {work / 'speak_r0.csv'}, {work / 'speak_r1.csv'}; {len(redo)} segments changed "
          f"(speak_r*_redo.csv): {' '.join(sorted(redo))}")


def embed_files(paths, work):
    """path -> (emotion2vec embedding, class scores), through emotion_match.py in the seedvc env."""
    lst = work / "_emo_list.txt"
    lst.write_text("\n".join(paths), encoding="utf-8")
    subprocess.run([str(ROOT / "envs" / "seedvc" / "Scripts" / "python.exe"), str(Path(__file__).with_name("emotion_match.py")),
                    "--embed-files", str(lst), "--embed-out", str(work / "_emo.npz")], check=True, cwd=str(Path(__file__).parent),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    z = np.load(work / "_emo.npz")
    return {str(p): (f, s) for p, f, s in zip(z["paths"], z["feats"], z["scores"])}


def measure(rows, work, train):
    """Per take of every segment: timbre, PESQ, CER, start/end shape against the performance, her ending habit
    (last segment), emotion2vec (centered cosine and category against the performance piece) and the arousal-cue
    distance to it (loudness left out, as in tts_eval.py). The slow part of pick: cached in <work>/_measures.json by
    file stamp, so a re-pick with other weights takes seconds."""
    cache_p = work / "_measures.json"
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}
    stamp = lambda p: f"{p.stat().st_size}:{int(p.stat().st_mtime)}"  # noqa: E731
    jobs = []
    for r in rows:
        sk = f"{r['key']}_s{r['seg']}"
        for t in takes_of(r, work):
            jobs.append((r, t))
    todo = [(r, t) for r, t in jobs if cache.get(str(t), {}).get("stamp") != stamp(t)]
    if todo:
        import torch
        import torchaudio
        from prep_voice import ecapa, embed
        from tts_eval import cer
        from tts_pick import ending_bank, ending_score, final_shape
        from vl_common import transcribe

        spk = ecapa()
        cen = np.mean([embed(spk, clip_path(r["id"])) for r in random.Random(1).sample(train, min(300, len(train)))], 0)
        cen /= np.linalg.norm(cen)
        asr = whisper_model()
        squim = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().eval()
        ebank = ending_bank(train)
        shapes = {}
        for k, (r, t) in enumerate(todo, 1):
            if r["perf"] not in shapes:
                px, psr = read_mono(r["perf"])
                shapes[r["perf"]] = (shape(px, psr, "start"), shape(px, psr, "end"))
            x, sr = read_mono(t)
            with torch.inference_mode():
                q = float(squim(torch.from_numpy(np.ascontiguousarray(prosody._resample(x, sr), dtype=np.float32))[None])[1])
            d = []
            for part, ref in zip(("start", "end"), shapes[r["perf"]]):
                s_ = shape(x, sr, part)
                if ref is not None and s_ is not None:
                    d.append(float(np.sqrt(np.mean((s_ - ref) ** 2))))
            last = int(r["seg"]) == int(r["n_seg"]) - 1
            cache[str(t)] = {"stamp": stamp(t), "timbre": float(embed(spk, t) @ cen), "pesq": q,
                             "cer": cer(r["text"], transcribe(asr, t)[0]), "shape": -float(np.mean(d)) if d else 0.0,
                             "ending": float(ending_score(final_shape(t), r["text"], ebank)) if last else 0.0}
            if k % 50 == 0:
                print(f"  measured {k}/{len(todo)}", flush=True)
                cache_p.write_text(json.dumps(cache), encoding="utf-8")
    # emotion and arousal: one emotion2vec pass over the takes and the performance pieces (seedvc env)
    need = [(r, t) for r, t in jobs if "emo_cos" not in cache[str(t)]]
    if need:
        from tts_eval import emotions

        emo = emotions(sorted({str(t) for _, t in need} | {str(Path(r["perf"])) for r, _ in need}), work)
        mean_bank = np.load(TARGET / "emo_bank.npz")["feats"].mean(0)
        num = lambda v: float(v) if v not in ("", None) else np.nan  # noqa: E731
        sd = np.nanstd(np.array([[num(x[k]) for k in AROUSAL] for x in read_csv(TARGET / "arousal_bank.csv")]), 0) + 1e-9
        w = np.array([0.0 if k == "level" else v for k, v in AROUSAL.items()])
        perf_ar = {}
        for r, t in need:
            fp, lp = emo[str(Path(r["perf"]))]
            ft, lt = emo[str(t)]
            cp, ct = fp - mean_bank, ft - mean_bank
            if r["perf"] not in perf_ar:
                perf_ar[r["perf"]] = arousal_raw(Path(r["perf"]), r["text"])
            ap_, at = perf_ar[r["perf"]], arousal_raw(t, r["text"])
            dist = 9.0
            if ap_ and at:
                dd = np.array([num(at[k]) - num(ap_[k]) for k in AROUSAL]) / sd
                ok = np.isfinite(dd) & (w > 0)
                dist = float(np.sqrt(np.sum(w[ok] * dd[ok] ** 2) / max(w[ok].sum(), 1e-9)))
            cache[str(t)].update({"emo_cos": float(cp @ ct / (np.linalg.norm(cp) * np.linalg.norm(ct) + 1e-9)),
                                  "emo_same": float(lt == lp), "emo_cat": lt, "perf_cat": lp, "arousal": dist})
    cache_p.write_text(json.dumps(cache), encoding="utf-8")
    return {str(t): cache[str(t)] for _, t in jobs}


def pick(a):
    from tts_pick import z as zs

    work, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(work / "segments.csv")
    hold = lines_of(a.set)[0][3]
    train = [r for r in read_csv(TARGET / "train.csv") if r["id"] not in hold]
    m = measure(rows, work, train)
    chosen, report = {}, []
    for r in rows:
        sk = f"{r['key']}_s{r['seg']}"
        takes = takes_of(r, work)
        if not takes:
            continue
        v = lambda k: np.array([m[str(t)][k] for t in takes])  # noqa: E731
        last = int(r["seg"]) == int(r["n_seg"]) - 1
        score = (2 * zs(v("timbre")) + zs(v("pesq")) - 30 * np.maximum(0, v("cer") - 0.1) + a.w_shape * zs(v("shape"))
                 + (a.w_ending * zs(v("ending")) if last else 0)
                 + a.w_emotion * (zs(v("emo_cos")) + a.w_cat * v("emo_same")) - a.w_arousal * zs(v("arousal")))
        k = int(np.argmax(score))
        chosen[sk] = takes[k]
        mk = m[str(takes[k])]
        report.append({"segment": sk, "text": r["text"], "pick": takes[k].parent.name + "/" + takes[k].name,
                       "timbre": round(mk["timbre"], 3), "pesq": round(mk["pesq"], 2), "cer": round(mk["cer"], 2),
                       "shape": round(mk["shape"], 2), "emo_cos": round(mk["emo_cos"], 2),
                       "emo": f"{mk['emo_cat']} / {mk['perf_cat']}", "arousal": round(mk["arousal"], 2)})
        print(f"{sk} -> {takes[k].parent.name}/{takes[k].name}", flush=True)
    write_csv(out / "pick.csv", report)
    # stitch at the performance's pauses and relative loudness
    by_line = {}
    for r in rows:
        by_line.setdefault(r["key"], []).append(r)
    for key, segs in by_line.items():
        sr = 32000
        perf_rms = []
        for r in segs:
            px, psr = read_mono(r["perf"])
            perf_rms.append(float(np.sqrt(np.mean(px ** 2)) + 1e-9))
        ref_level = max(perf_rms)
        pieces = []
        for i, r in enumerate(segs):
            t = chosen.get(f"{key}_s{r['seg']}")
            if t is None:
                continue
            x, sr = read_mono(t, sr)
            x = trim_silence(x, sr)
            x = x * (perf_rms[i] / ref_level) / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 0.1
            fade = min(int(0.005 * sr), len(x) // 2)
            x[:fade] *= np.linspace(0, 1, fade)
            x[len(x) - fade:] *= np.linspace(1, 0, fade)
            pieces.append(x)
            if i < len(segs) - 1:
                pieces.append(np.zeros(int((float(r["next"]) - float(r["end"])) * sr), dtype=np.float32))
        y = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
        y = y * (0.9 / np.abs(y).max()) if np.abs(y).max() > 0.9 else y
        sf.write(out / f"{key}.wav", y, sr, subtype="PCM_16")
    print(f"{len(by_line)} lines -> {out}")


def carve(src, dst, r):
    """The short piece out of a take spoken with its carrier (plan: 'say' = carrier + piece, or piece + carrier): the
    pause nearest where the piece should begin or end (by its share of the characters), longer pauses preferred."""
    import librosa

    x, sr = read_mono(src)
    ps = [p for p in pauses(x, sr) if p[1] - p[0] >= 0.06]
    if not ps:
        return False
    chars = lambda t: len(re.sub(r"[、,，。?？!！…\s]", "", t))  # noqa: E731
    share = chars(r["text"]) / max(1, chars(r["say"]))
    _, (a, b) = librosa.effects.trim(x, top_db=30)
    t_exp = (a + (1 - share if r["carrier"] == "before" else share) * (b - a)) / sr
    p0, p1 = min(ps, key=lambda p: abs((p[0] + p[1]) / 2 - t_exp) - (p[1] - p[0]))
    y = trim_silence(x[int(p1 * sr):] if r["carrier"] == "before" else x[:int(p0 * sr)], sr)
    if not 0.2 <= len(y) / sr <= 3.0:
        return False
    sf.write(dst, y, sr, subtype="PCM_16")
    return True


def takes_of(r, work):
    """A segment's takes; for a piece spoken with a carrier, the piece carved out of each (<work>/carved/)."""
    sk = f"{r['key']}_s{r['seg']}"
    takes = sorted(t for d in sorted(work.glob("takes_r*")) for t in d.glob(f"{sk}_t*.wav"))
    if not r.get("carrier"):
        return takes
    out = []
    for t in takes:
        c = work / "carved" / t.parent.name / t.name
        if not c.exists() or c.stat().st_mtime < t.stat().st_mtime:
            c.parent.mkdir(parents=True, exist_ok=True)
            if not carve(t, c, r):
                continue
        out.append(c)
    return out


def trim_silence(x, sr):
    import librosa

    _, (a, b) = librosa.effects.trim(x, top_db=40)
    return x[a:b].copy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plan")
    s.add_argument("--set", default="test")
    s.add_argument("--work", required=True)
    s.set_defaults(fn=plan)
    s = sub.add_parser("pick")
    s.add_argument("--set", default="test")
    s.add_argument("--work", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--w-shape", type=float, default=1.5)
    s.add_argument("--w-ending", type=float, default=1.0)
    s.add_argument("--w-emotion", type=float, default=1.0, help="emotion2vec centered cosine to the performance piece")
    s.add_argument("--w-cat", type=float, default=0.5, help="bonus for the performance piece's emotion2vec category")
    s.add_argument("--w-arousal", type=float, default=1.0, help="arousal-cue distance to the performance piece")
    s.set_defaults(fn=pick)
    a = ap.parse_args()
    a.fn(a)
