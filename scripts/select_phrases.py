"""Round 4: pick the best sampled take per phrase and stitch the lines back together (tools env).

Each candidate from run_indextts.py --phrases is scored against the original phrase:
  corr     intonation follows the original (prosody.contour_corr)        higher is better
  cer      Whisper transcript vs the phrase text                         a misread costs 2x
  dev      speech length vs the original phrase                          10 % costs 0.1
  reg      semitones away from the original register shifted to the target voice's     1 st costs 0.1, beyond 2 st 0.5
  score = corr - 2*cer - dev - 0.1*reg - 0.4*max(0, reg - 2)
Then per line, the phrases have to fit together: a phrase's pitch relative to the line's other chosen
phrases should sit where the original phrase sits relative to its line (step); beyond 2 st each semitone
costs 0.3 more, and each phrase is re-chosen against its neighbours (twice). Stitched lines that sounded
off-key had one phrase several semitones out of line while every take scored well on its own.
The chosen takes are placed where the original phrases start (pauses and timing stay the
original's), each scaled to the original phrase's loudness relative to the line, at 22.05 kHz.

Writes exp/<name>/Lxx.wav, exp/<name>/selection.csv. Re-runs reuse the scores of unchanged takes and
leave unchanged lines untouched (stitch.json), so the later steps skip them too.
Usage:  envs/tools/Scripts/python scripts/select_phrases.py --cands D4_cands --name D4_indextts_phrase
"""
import argparse
import json
import math
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

import prosody
from vl_common import EXP, PHRASES, TARGET, load_lines, normalize_ja, read_csv, read_mono, transcribe, \
    whisper_model, write_csv
from vl_paths import routes, voice_cfg

SR = 22050
FADE = int(0.005 * SR)


def trim(wav, sr):
    _, (a, b) = librosa.effects.trim(wav, top_db=40)
    return wav[a:b]


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))) + 1e-9)


def cer(ref, hyp):
    import jiwer

    r = normalize_ja(ref)
    return float(jiwer.cer(r, normalize_ja(hyp))) if r else 0.0


def register_offset():
    """The voice's register relative to the game character's, in semitones (same estimate as metrics.f0_toward_target).
    Against all of the character's lines; a voice with `register_lines: "speech"` in its settings is measured
    against the lines that are re-dubbed (routes speech / refine) - grunts and cries sit ~3 st above the speech
    (2026-10-14: source 3.1, other 2.9) and pulled every phrase's target that far down. The voices made before
    keep the old estimate, so their picks do not move."""
    zt = sorted((TARGET / "refs").glob("abx_*.wav")) + [TARGET / "refs" / "ref_12s.wav"]
    st_02 = np.nanmedian([prosody.median_st(*read_mono(p)) for p in zt])
    lines = load_lines()
    if voice_cfg().get("register_lines") == "speech":
        rt = routes()
        lines = [ln for ln in lines if rt.get(ln["id"]) in ("speech", "refine")] or lines
    st_hy = np.nanmedian([prosody.median_st(*read_mono(ln["wav"])) for ln in lines])
    return float(st_02 - st_hy)


def score_take(asr, path, text, o_sp, o_sr, offset):
    """Score one take against the original phrase's voiced part (see module docstring)."""
    wav, sr = read_mono(path)
    sp = trim(wav, sr)
    hyp, _, _ = transcribe(asr, path)
    o_span, o_reg = len(o_sp) / o_sr, prosody.median_st(o_sp, o_sr)
    med = prosody.median_st(sp, sr)
    m = {"corr": prosody.contour_corr(o_sp, o_sr, sp, sr), "cer": cer(text, hyp),
         "dev": abs(len(sp) / sr - o_span) / o_span, "reg": abs(med - (o_reg + offset))}
    m = {k: (0.0 if k == "corr" else 1.0) if math.isnan(v) else v for k, v in m.items()}
    m["med"] = med
    m["score"] = base_score(m)
    return m, hyp


REG_KNEE, REG_STEEP, STEP_KNEE, STEP_COST = 2.0, 0.4, 2.0, 0.3


def base_score(m):
    return m["corr"] - 2 * m["cer"] - m["dev"] - 0.1 * m["reg"] - REG_STEEP * max(0.0, m["reg"] - REG_KNEE)


def fit_lines(phrases, takes, o_med):
    """Re-choose each phrase against its neighbours: takes = pid -> [(metrics, path)], o_med = pid -> original
    phrase median (st). Returns pid -> (metrics, path, step)."""
    by_line = {}
    for p in phrases:
        by_line.setdefault(p["line_id"], []).append(p["id"])
    pick = {pid: max(ts, key=lambda t: t[0]["score"]) for pid, ts in takes.items()}
    step_of = {pid: 0.0 for pid in takes}
    for pids in by_line.values():
        if len(pids) < 2:
            continue
        for _ in range(2):
            for pid in pids:
                others = [q for q in pids if q != pid and np.isfinite(pick[q][0]["med"]) and np.isfinite(o_med[q])]
                if not others or not np.isfinite(o_med[pid]):
                    continue
                ref_t = np.median([pick[q][0]["med"] for q in others])
                ref_o = np.median([o_med[q] for q in others])

                def total(t):
                    if not np.isfinite(t[0]["med"]):
                        return t[0]["score"], 0.0
                    step = (t[0]["med"] - ref_t) - (o_med[pid] - ref_o)
                    return t[0]["score"] - STEP_COST * max(0.0, abs(step) - STEP_KNEE), step

                best = max(takes[pid], key=lambda t: total(t)[0])
                pick[pid], step_of[pid] = best, total(best)[1]
    return {pid: (t[0], t[1], step_of[pid]) for pid, t in pick.items()}


def place(phrases, takes, line_wav, sr=SR):
    """Stitch takes (pid -> path) on the original timing, each at the original phrase's relative level."""
    line_rms = rms(trim(line_wav, sr))
    out = np.zeros(len(line_wav), dtype=np.float32)
    prev_end = 0
    for p in phrases:
        seg = trim(read_mono(takes[p["id"]], sr)[0], sr)
        o_seg = line_wav[int(float(p["speech_start"]) * sr):int(float(p["speech_end"]) * sr)]
        seg = seg * (rms(o_seg) / line_rms) / (rms(seg) / 0.1)  # keep the original phrase-to-phrase dynamics
        n = min(FADE, len(seg) // 2)
        seg[:n] *= np.linspace(0, 1, n)
        seg[len(seg) - n:] *= np.linspace(1, 0, n)
        start = max(int(float(p["speech_start"]) * sr), prev_end + int(0.06 * sr))
        if start + len(seg) > len(out):
            out = np.pad(out, (0, start + len(seg) - len(out) + int(0.05 * sr)))
        out[start:start + len(seg)] += seg
        prev_end = start + len(seg)
    peak = float(np.abs(out).max())
    return out * (0.9 / peak) if peak > 0.9 else out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default="D4_cands", help="exp/ folder with <phrase>_c<k>.wav + cands.json")
    ap.add_argument("--name", default="D4_indextts_phrase")
    ap.add_argument("--phrases-dir", default=str(PHRASES), help="phrases.csv + phrase wavs (default: the set's)")
    ap.add_argument("--legacy", action="store_true",
                    help="the scoring before 2026-10-12 (0.1/st register only, no line fit): reproduces older results")
    a = ap.parse_args()
    if a.legacy:
        REG_STEEP, STEP_COST = 0.0, 0.0

    cand_dir, out_dir = EXP / a.cands, EXP / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    phr_dir = Path(a.phrases_dir)
    phrases = read_csv(phr_dir / "phrases.csv")
    info = {p["id"]: p for p in json.loads((cand_dir / "cands.json").read_text(encoding="utf-8"))["phrases"]}
    offset = register_offset()
    print(f"{TARGET.name} register offset: {offset:+.2f} semitones")

    # scores of takes that have not changed since the last run are reused (re-runs redo only new takes)
    stamp = lambda p: f"{p.stat().st_size}-{p.stat().st_mtime_ns}"  # noqa: E731
    cache, sel_p = {}, out_dir / "selection.csv"
    if sel_p.exists():
        for r in read_csv(sel_p):
            if not r.get("score"):
                continue
            if not r.get("stamp"):  # written before stamps existed: valid if the take predates that run
                take = cand_dir / r["cand"]
                if not take.exists() or take.stat().st_mtime > sel_p.stat().st_mtime:
                    continue
                r["stamp"] = stamp(take)
            cache[(r["phrase"], r["cand"], r["stamp"])] = r
    asr = None
    takes, o_med, asr_text = {}, {}, {}
    for p in phrases:
        o_wav, o_sr = read_mono(phr_dir / f"{p['id']}.wav")
        o_sp = trim(o_wav, o_sr)
        o_med[p["id"]] = prosody.median_st(o_sp, o_sr)
        takes[p["id"]] = []
        for c in info[p["id"]]["cands"]:
            path = cand_dir / c["file"]
            hit = cache.get((p["id"], c["file"], stamp(path)))
            if hit:
                m, text = {k: float(hit[k]) for k in ("corr", "cer", "dev", "reg")}, hit["asr"]
                m["med"] = float(hit["med"]) if hit.get("med") not in (None, "") else \
                    prosody.median_st(trim(*read_mono(path)), read_mono(path)[1])
                # the register term follows today's offset: a cached one may come from another estimate
                reg = abs(m["med"] - (o_med[p["id"]] + offset))
                m["reg"] = 1.0 if math.isnan(reg) else reg
                m["score"] = base_score(m)
            else:
                asr = asr or whisper_model()
                m, text = score_take(asr, path, p["text"], o_sp, o_sr, offset)
            takes[p["id"]].append((m, path))
            asr_text[path.name] = text
    picked = fit_lines(phrases, takes, o_med)
    chosen, report = {}, []
    for p in phrases:
        for m, path in takes[p["id"]]:
            report.append({"phrase": p["id"], "cand": path.name, "asr": asr_text[path.name], "stamp": stamp(path),
                           **{k: round(v, 3) if np.isfinite(v) else "" for k, v in m.items()}})
        m, path, step = picked[p["id"]]
        report.append({"phrase": p["id"], "cand": f"-> {path.name}", "step": round(step, 2)})
        chosen[p["id"]] = path
        print(f"{p['id']} -> {path.name}  step={step:+.2f} " + " ".join(f"{k}={v:.2f}" for k, v in m.items()))
    write_csv(out_dir / "selection.csv", report,
              fieldnames=["phrase", "cand", "score", "corr", "cer", "dev", "reg", "med", "step", "asr", "stamp"])

    # a line is re-stitched only when its takes or timing changed, so later steps can skip it as well
    sig_p = out_dir / "stitch.json"
    sigs = json.loads(sig_p.read_text(encoding="utf-8")) if sig_p.exists() else {}
    done, written = {p["line_id"] for p in phrases}, 0  # non-verbal lines have no phrases (routes)
    for ln in [ln for ln in load_lines() if ln["id"] in done]:
        mine = [p for p in phrases if p["line_id"] == ln["id"]]
        sig = "|".join(f"{chosen[p['id']].name}:{stamp(chosen[p['id']])}:{p['speech_start']}" for p in mine)
        dst = out_dir / f"{ln['id']}.wav"
        if sigs.get(ln["id"]) == sig and dst.exists():
            continue
        out = place(mine, chosen, read_mono(ln["wav"], SR)[0])
        sigs[ln["id"]] = sig
        if dst.exists():  # same audio as before (first run with stitch.json): keep the file and its mtime
            old, _ = sf.read(str(dst), dtype="float32")
            if len(old) == len(out) and np.allclose(old, out, atol=2 / 32768):
                continue
        sf.write(dst, out, SR, subtype="PCM_16")
        written += 1
    sig_p.write_text(json.dumps(sigs, indent=1), encoding="utf-8")
    print(f"stitched {written} new/changed of {len(done)} lines -> {out_dir}")
