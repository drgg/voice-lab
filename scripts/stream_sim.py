"""Streaming simulation of the recording replacement (tools env): what a real-time system whose output is
delayed by L ms would have done on a recording.

Real-time chain in mind: game audio -> virtual cable -> this processing -> speakers, all delayed by L.
A line is replaced from its first sample only if it is recognised within L ms of its onset (minus PROC for
the audio block and compute). Recognised later, the replacement cuts in late and the first part goes out in
the original voice. Accepted wrongly, a piece of the wrong target line plays until the error shows.

For each L, every candidate (100 ms prefix matched filter) is first judged when L - PROC ms of it have been
heard (waiting costs nothing inside the delay), then at later checkpoints while undecided. Evidence and
rules are the offline ones (replace_in_recording.verify / strong) on the part heard so far, computed on what
is left after the lines already accepted (claims). The character speaks one line at a time: a newly
accepted line ends the one before it.

Reference: the offline result (events.csv). Writes exp/<set>/recordings/<name>/stream/{stream.csv, L<ms>.wav}
and the page eval/<set>/stream_<name>.html.
Usage:  envs/tools/Scripts/python scripts/stream_sim.py 2026-10-10_00-37-31 [--signal mix] [--L 150,200,300,500,800]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, find_peaks, resample_poly, sosfilt

import replace_in_recording as R
from rt_feasibility import onset
from vl_common import EVAL, EXP, load_lines, read_csv, read_mono, write_csv
from vl_paths import EXP_BASE, page_path, voice_cfg

SR16, SR = R.SR16, R.SR
PROC = 20                  # ms: one 10 ms audio block plus compute
CK = [100, 150, 200, 300, 400, 600, 1000, 1500]  # ms heard at which undecided candidates are re-judged
TRACK_W, TRACK_HOP = 0.2, 0.05   # s: windows in which a playing line's continuation is followed
CUT_NCC, CUT_RUN = 0.15, 0.5     # a line counts as stopped once it matches below CUT_NCC for CUT_RUN s on end:
                                 # a real stop drops to ~0 at once and stays there, while effects can bury a
                                 # line that keeps playing for ~0.3 s
CUT_FAST, CUT_DROP = 0.1, 8.0    # ...but when the band also falls CUT_DROP dB below what the line itself predicts
                                 # there (a quiet scene where the voice simply stops), CUT_FAST s is enough
PREFIXES, PREFIX_THR = (0.1, 0.2), 0.45  # s of real sound matched to make candidates: 100 ms catches most lines
                                         # early; short grunts (はっ, ふん) only match after 150-200 ms
OVERRIDE = 0.2             # how much better a candidate must match to interrupt a line still playing
CONT = 0.25                # a playing line still matching this well is not interrupted: once the game cuts a line its
                           # continuation drops to ~0, while its quiet stretches under effects still score 0.4-0.5


def rules(ex, nl, ncc, heard, voiced_line, short, rule):
    """offline: replace_in_recording.strong() on the heard part (heard >= 1 s stands in for "a long line").
    Otherwise rule = (min explained - null, min ncc, min explained): on the mix the waveform match of the
    right line is lower than on the stem (median 0.45 after 300 ms), so the energy test carries more."""
    if rule is None and short:
        return ex - nl >= 3.0 and ex >= 10.0 and ncc >= 0.55
    if rule is None:
        if ex - nl < 3.0:
            return False
        return ((ex >= 8.0 and ncc >= 0.45) or (ex >= 6.0 and ncc >= 0.6) or
                (ex >= 5.0 and ncc >= 0.5 and min(heard, voiced_line) >= 1.0))
    dn, nc, mx = rule
    return ex - nl >= dn and ncc >= nc + (0.05 if short else 0.0) and ex >= mx  # breaths match more by chance


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="recording folder under exp/<set>/recordings/")
    ap.add_argument("--signal", default="mix", choices=["mix", "stem"],
                    help="mix: what a real-time chain has; stem: offline separation (optimistic bound)")
    ap.add_argument("--L", default="150,200,300,500,800", help="output delays to simulate, ms")
    ap.add_argument("--final", default="final_48k")
    ap.add_argument("--rule", default="", help="min explained-null, min ncc, min explained "
                    "(default: 3,0.7,4 on the mix, the offline rules on the stem)")
    ap.add_argument("--prefix-thr", type=float, default=PREFIX_THR, help="100 ms prefix score of a candidate")
    ap.add_argument("--tag", default="", help="suffix of the outputs, to compare rule sets")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--removal", default="mix", choices=["mix", "stem"],
                    help="what the removal is fitted on when rendering: mix (what real time has) or the offline stem")
    ap.add_argument("--path", default="auto", help="voice path FIR (.npy) for the waveform matching; auto: "
                    "<recording>/voice_path_fir.npy if present (replace_in_recording.py writes it); none")
    ap.add_argument("--debug", default="", help="t0,t1 (s): print every judgement of candidates starting there")
    a = ap.parse_args()
    # mix (with the voice path): waveform match carries it, energy only has to beat the null; stem: offline rules
    rule = tuple(float(v) for v in (a.rule or ("3,0.7,4" if a.signal == "mix" else "")).split(","))         if (a.rule or a.signal == "mix") else None

    work = EXP_BASE / "recordings" / a.name
    out = work / "stream"
    out.mkdir(exist_ok=True)
    lines = load_lines()
    ids = [ln["id"] for ln in lines]
    fs = {r["id"]: r["source"] for r in read_csv(EXP / a.final / "final.csv")}
    x16 = [read_mono(ln["wav"], SR16)[0] for ln in lines]
    on = [onset(x) for x in x16]
    x48 = [read_mono(ln["wav"], SR)[0].astype(np.float64) for ln in lines]
    # The game's voice path is an EQ with non-linear phase: a plain waveform match of the file swings with
    # whichever band dominates (even negative while the line plays). Matching templates are passed through
    # the path, calibrated once from lines heard in a quiet scene (the block fits need none of this).
    fir = work / "voice_path_fir.npy" if a.path == "auto" else (Path(a.path) if a.path != "none" else None)
    if fir is not None and fir.exists():
        h = np.load(fir)
        xm = [fftconvolve(x, h)[len(h) // 2:len(h) // 2 + len(x)] for x in x48]
        x16 = [resample_poly(x, 1, 3).astype(np.float32) for x in xm]
        print(f"matching through the voice path {fir.name} ({len(h)} taps)")
    else:
        xm = x48
        print("matching the files as they are (no voice path)")
    b48 = [sosfilt(R.BAND48, x) for x in xm]
    loud = [np.flatnonzero(np.abs(x) > 0.05 * np.abs(x).max()) for x in x48]
    voiced_line = [(lo[-1] - on[k] * 3) / SR for k, lo in enumerate(loud)]       # s from onset to last sound
    short = [len(lo) / SR < R.SHORT for lo in loud]
    P = np.stack([np.pad(f[o:o + 4800], (0, max(0, 4800 - len(f[o:o + 4800]))))
                  for f, o in zip((R.feat(x) for x in x16), on)])
    nP = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    # > 0.9: the same audio in two files (also when their leading silences differ: lags up to +-20 ms)
    G = int(0.02 * SR16)
    nfft = 1 << int(np.ceil(np.log2(P.shape[1] + 2 * G)))
    FP = np.fft.rfft(np.pad(nP, ((0, 0), (G, G))), nfft, axis=1)
    FQ = np.conj(np.fft.rfft(nP, nfft, axis=1))
    SIM = np.stack([np.abs(np.fft.irfft(FP[i][None] * FQ, nfft, axis=1)[:, :2 * G + 1]).max(axis=1)
                    for i in range(len(nP))])
    SIM = np.maximum(SIM, SIM.T)
    ref = [(e["id"], int(round(float(e["time"]) * SR)) + on[ids.index(e["id"])] * 3, e["version"])
           for e in read_csv(work / "events.csv")]

    game = work / ("game_single.wav" if (work / "game_single.wav").exists() else "game.wav")
    mix = sf.read(str(game), dtype="float32", always_2d=True)[0]
    stem = work / ("stems_single" if game.name == "game_single.wav" else "stems") / "vocals.wav"
    voc = R.fit_len(sf.read(str(stem), dtype="float32", always_2d=True)[0], len(mix))
    S48 = (mix if a.signal == "mix" else voc).mean(axis=1).astype(np.float64)

    # candidates: matched filter of the first 100 ms of real sound (grunts often start with a quiet lead-in that
    # matches nothing), best peak per line within 25 ms; a candidate exists only once that much was heard
    F = R.feat(R.to16(mix if a.signal == "mix" else voc))
    cands, avail = [], []
    for k in range(len(lines)):
        if len(loud[k]) < R.MIN_VOICED * SR:
            continue
        st = max(on[k], loud[k][0] // 3)  # 16 kHz
        seen = []
        for pre in PREFIXES:
            t = R.feat(x16[k])[st:st + int(pre * SR16)]
            if np.linalg.norm(t) < 1e-4:
                continue
            r = R.ncc_series(F, t)
            for p in find_peaks(r, height=a.prefix_thr, distance=int(0.025 * SR16))[0]:
                if any(abs(p - q) < 0.025 * SR16 for q in seen):
                    continue  # already a candidate from the shorter prefix
                seen.append(p)
                cands.append((k, (int(p) - (st - on[k])) * 3))  # onset, 48 kHz
                avail.append((st - on[k]) / SR16 * 1000 + pre * 1000)  # ms after the onset
    print(f"{a.signal}: {len(cands)} candidates")

    ALIGN = int(0.03 * SR)

    def evidence(sig, sig_b, k, o48, heard):
        """(ncc, explained, null, played samples from file start, onset) over `heard` ms from the onset. The
        onset is first re-aligned within +-30 ms on all that was heard: the 100 ms prefix often peaks a pitch
        period off, which the block fit forgives but the waveform match and the continuation check do not."""
        n = min(on[k] * 3 + int(heard * SR / 1000), len(x48[k]))
        t = b48[k][on[k] * 3:n]
        for _ in range(4):  # re-centre until the best match stops moving (it can sit near an edge)
            lo = o48 - ALIGN
            if lo < on[k] * 3 or lo + len(t) + 2 * ALIGN > len(sig):
                return None
            r = R.ncc_series(sig_b[lo:lo + len(t) + 2 * ALIGN], t)
            i = int(np.argmax(r))
            moved = abs(lo + i - o48) > 2
            o48, ncc = lo + i, float(r[i])
            if not moved:
                break
        else:
            return ncc, 0.0, 0.0, 0, o48  # never settled: no stable alignment, not this line here
        s = o48 - on[k] * 3
        if ncc < min(0.4, rule[1] if rule else 0.4):  # no rule can pass; skip the costly fit
            return ncc, 0.0, 0.0, 0, o48
        ex, nl, played = R.verify(sig, x48[k], s, n)
        return ncc, ex, nl, played, o48

    S48_b = sosfilt(R.BAND48, S48)

    dbg = tuple(float(v) * SR for v in a.debug.split(",")) if a.debug else None

    def say(o48, msg):
        if dbg and dbg[0] <= o48 <= dbg[1]:
            print(f"    {o48 / SR:7.3f}s {msg}")

    def continuation(acc, o48, heard):
        """Best match, over what the candidate heard, of a line already accepted and still playing at o48
        (claims, done the online way: subtracting a copy instead lets an early wrong pick eat the right line
        that follows). Returns (match, how well that line matched when it was accepted)."""
        n = int(heard * SR / 1000)
        best = (-1.0, 0.0)
        for b in acc:
            if not b["o48"] < o48 < b["s"] + b["played"]:
                continue
            i = on[b["k"]] * 3 + (o48 - b["o48"])
            t = b48[b["k"]][i:i + n]
            if len(t) < n // 2 or np.linalg.norm(t) < 0.1 * np.linalg.norm(b48[b["k"]][on[b["k"]] * 3:][:n]):
                continue  # the line is over or in a pause there
            seg = S48_b[o48:o48 + len(t)]
            c = float(np.dot(seg, t) / (np.linalg.norm(seg) * np.linalg.norm(t) + 1e-9))
            say(o48, f"continuation of {ids[b['k']]} there: ncc {c:.2f}")
            best = max(best, (c, b["ncc"]))
        return best

    def track(k, o48):
        """How long an accepted line plays, as a streaming tracker would see it on the signal: to the end of
        its file unless its continuation stops matching for CUT_RUN s (the game cut it). Returns (played
        samples from the file start, ms after the cut point at which the tracker knows)."""
        s, W, H = o48 - on[k] * 3, int(TRACK_W * SR), int(TRACK_HOP * SR)
        ref = np.linalg.norm(b48[k][on[k] * 3:on[k] * 3 + W])
        run_start, run, level = None, 0.0, []  # (signal rms, line rms) of matching windows
        for i in range(on[k] * 3, len(x48[k]) - W // 2, H):
            t = b48[k][i:i + W]
            seg = S48_b[s + i:s + i + len(t)]
            if len(seg) < len(t):
                break
            if np.linalg.norm(t) < 0.05 * ref:
                continue  # a pause in the line: no evidence either way
            c = np.dot(seg, t) / (np.linalg.norm(seg) * np.linalg.norm(t) + 1e-9)
            rms, trms = float(np.sqrt(np.mean(seg ** 2)) + 1e-12), float(np.sqrt(np.mean(t ** 2)) + 1e-12)
            if c < CUT_NCC:
                run_start = i if run_start is None else run_start
                run += TRACK_HOP
                # how far the signal fell, beyond what the line's own soft passage would explain
                quiet = bool(level) and 20 * np.log10(np.median([l[0] / l[1] for l in level]) / (rms / trms)) >= CUT_DROP
                say(o48, f"track {ids[k]} at {(i - on[k] * 3) / SR:.2f}s: ncc {c:.2f} run {run:.2f}s quiet {quiet} "
                         f"drop {20 * np.log10(np.median([l[0] / l[1] for l in level]) / (rms / trms)) if level else 0:.1f}")
                if run >= CUT_RUN or (quiet and run >= CUT_FAST):
                    return run_start, (i + W - run_start) / SR * 1000
            else:
                run_start, run = None, 0.0
                level.append((rms, trms))
        return len(x48[k]), 0.0

    Ls = [int(v) for v in a.L.split(",")]
    summary, pages = [], {}
    for L in Ls:
        first = L - PROC
        # queue of (time the judgement can be made, candidate, heard ms)
        queue = []
        for c, (k, o48) in enumerate(cands):
            full = voiced_line[k] * 1000
            h0 = max(min(first, full) if full > 0 else first, avail[c])
            queue.append((o48 + int(h0 * SR / 1000), c, h0))
        queue.sort()
        acc, done = [], set()
        qi = 0
        while qi < len(queue):
            T, c, heard = queue[qi]
            qi += 1
            if c in done:
                continue
            k, o48 = cands[c]
            if any(abs(o48 - b["o48"]) < 0.03 * SR and SIM[k, b["k"]] >= 0.9 for b in acc):
                done.add(c)
                continue  # the same audio is already accepted
            ev = evidence(S48, S48_b, k, o48, heard)
            if ev is None:
                done.add(c)
                continue
            ncc, ex, nl, played, o48 = ev
            cands[c] = (k, o48)  # keep the refined onset for later judgements
            if any(abs(o48 - b["o48"]) < 0.03 * SR and SIM[k, b["k"]] >= 0.9 for b in acc):
                done.add(c)
                continue  # converged onto a line already accepted
            cont, cont_acc = continuation(acc, o48, heard)
            if cont >= CONT and (ncc < cont + OVERRIDE or ncc < cont_acc + 0.05):
                # a stretch of the line already playing. A real interruption shows as that line no longer
                # matching (cont < CONT); a wrong line accepted just before the right one is overridden
                # because the right one matches clearly better, also better than the wrong one ever did.
                done.add(c)
                continue
            ok = played and rules(ex, nl, ncc, heard / 1000, voiced_line[k], short[k], rule)
            say(o48, f"L={L} {ids[k]} heard {heard:.0f} ms: ncc {ncc:.2f} ex {ex:.1f} null {nl:.1f} played "
                     f"{played / SR:.2f}s -> {'ACCEPT' if ok else 'wait/reject'}")
            if ok:
                done.add(c)
                b = {"k": k, "o48": o48, "s": o48 - on[k] * 3, "D": heard + PROC, "ex": ex, "ncc": ncc}
                # the character speaks one line at a time: the newest line ends any still playing, including a
                # replacement still sounding because its stop was not yet noticed (CUT_RUN lag)
                for prev in acc:
                    lag = 0 if prev.get("cut_by") else int(max(0, prev["cut_det"] - L) * SR / 1000)
                    if prev["o48"] < b["o48"] < prev["s"] + prev["played"] + lag:
                        prev["played"] = min(prev["played"], b["o48"] - prev["s"])
                        prev["cut_by"] = b
                b["played"], b["cut_det"] = track(k, o48)
                acc.append(b)
                continue
            nxt = [h for h in CK if h > heard]
            full = voiced_line[k] * 1000
            if heard >= full or not nxt or (heard >= 150 and ncc < min(0.3, rule[1] - 0.05 if rule else 0.3)):
                done.add(c)
                continue
            h = min(nxt[0], full) if full > heard else nxt[0]
            T2 = o48 + int(h * SR / 1000)
            j = qi
            while j < len(queue) and queue[j][0] <= T2:
                j += 1
            queue.insert(j, (T2, c, h))

        # score against the offline reference
        rows, used = [], set()
        for b in sorted(acc, key=lambda b: b["o48"]):
            m = [i for i, (rid, p0, _) in enumerate(ref)
                 if abs(b["o48"] - p0) <= 0.03 * SR and SIM[b["k"], ids.index(rid)] >= 0.9 and i not in used]
            late = max(0, b["D"] - L)
            nxt = b.get("cut_by")
            over = max(0, nxt["D"] - L) if nxt else max(0, b["cut_det"] - L)
            kind = "correct" if m else "wrong"
            if m:
                used.add(m[0])
            rows.append({"L": L, "time": round(b["o48"] / SR, 2), "id": ids[b["k"]], "text": lines[b["k"]]["ref_text"],
                         "result": kind, "decided_ms": int(round(b["D"])), "late_ms": int(round(late)),
                         "overrun_ms": int(round(over)),
                         "ncc": round(b["ncc"], 2), "explained_db": round(b["ex"], 1),
                         "replaced": "" if fs.get(ids[b["k"]], "N_orig").startswith("N_orig") else "yes"})
            b["late"], b["over"], b["kind"] = late, over, kind
        miss = [{"L": L, "time": round(p0 / SR, 2), "id": rid, "text": lines[ids.index(rid)]["ref_text"],
                 "result": "missed", "decided_ms": "", "late_ms": "", "overrun_ms": "", "ncc": "", "explained_db": "",
                 "replaced": "" if v == "保留原声" else "yes"}
                for i, (rid, p0, v) in enumerate(ref) if i not in used]
        rows = sorted(rows + miss, key=lambda r: r["time"])
        rep = [r for r in rows if r["replaced"]]
        on_time = sum(r["result"] == "correct" and r["late_ms"] == 0 for r in rep)
        late = [r for r in rep if r["result"] == "correct" and r["late_ms"] > 0]
        wrong = [r for r in rep if r["result"] == "wrong"]
        wrong_kept = [r for r in rows if r["result"] == "wrong" and not r["replaced"]]
        missed = [r for r in rep if r["result"] == "missed"]
        n_ref = sum(1 for _, _, v in ref if v != "保留原声")
        late_s = ", ".join("%s+%sms" % (r["id"], r["late_ms"]) for r in late)
        miss_s = ", ".join(r["id"] for r in missed)
        wrong_s = ", ".join("%s@%s" % (r["id"], r["time"]) for r in wrong)
        print(f"L={L:4d} ms: from the start {on_time}/{n_ref}, late {len(late)} ({late_s}), "
              f"missed {len(missed)} ({miss_s}), wrong {len(wrong)} ({wrong_s}); "
              f"wrong picks of lines kept as the original (silent) {len(wrong_kept)}", flush=True)
        summary += rows

        if a.no_render:
            continue
        # render what the listener would hear (delay removed so it lines up with the recording)
        plan = []
        for b in sorted(acc, key=lambda b: b["o48"]):
            k = b["k"]
            if fs.get(ids[k], "N_orig").startswith("N_orig"):
                continue
            fin = read_mono(EXP / a.final / f"{ids[k]}.wav", SR)[0].astype(np.float64)
            end = b["played"] + int(b["over"] * SR / 1000)
            plan.append({"k": k, "s": b["s"], "x": x48[k][:b["played"]], "fin": R.fade_out(fin[:end]) if end < len(fin) else fin,
                         "cut": b["played"] < len(x48[k]) - int(0.5 * SR),
                         "start": on[k] * 3 + int(b["late"] * SR / 1000) if b["late"] else 0})
        removed, replaced, _, stats = R.render(mix, mix if a.removal == "mix" else voc, plan)
        peak = float(np.abs(removed).max())
        sf.write(out / f"L{L}{a.tag}_removed.wav", removed / peak * 0.99 if peak > 0.99 else removed, SR, subtype="PCM_16")
        drops = [float(np.mean(d)) for d, _ in stats if d]
        if drops:
            print(f"    removal fitted on the {a.removal}: voice drop median {np.median(drops):.1f} dB, "
                  f"min {min(drops):.1f} dB", flush=True)
        peak = float(np.abs(replaced).max())
        wav = out / f"L{L}{a.tag}.wav"
        sf.write(wav, replaced / peak * 0.99 if peak > 0.99 else replaced, SR, subtype="PCM_16")
        pages[L] = {"wav": page_path(wav), "on_time": on_time, "late": len(late), "missed": len(missed),
                    "wrong": len(wrong), "n": n_ref}

    write_csv(out / f"stream_{a.signal}{a.tag}.csv", summary)
    if a.no_render:
        raise SystemExit
    data = {"name": a.name, "signal": a.signal, "game": page_path(work / "game16.wav"),
            "offline": page_path(work / "replaced.wav"), "Ls": pages, "rows": summary}
    tpl = (Path(__file__).parent / "templates" / "stream.html").read_text(encoding="utf-8")
    page = EVAL / f"stream_{a.name}_{a.signal}{a.tag}.html"
    page.write_text(tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False, default=lambda o: o.item()))
                    .replace("__VOICE__", voice_cfg()["label"]),
                    encoding="utf-8")
    print(f"-> {page}")
