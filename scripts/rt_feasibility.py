"""Feasibility of replacing known lines in a game's audio output from outside the game (tools env).

The game plays exactly the exported originals, so a line can be spotted by matched filtering against
the original's first W ms (normalised cross-correlation), not by generic fingerprinting that needs
seconds. Once found to the sample, the original can be subtracted using itself as the reference.

Simulation on the current set: each line is mixed at a random point into anime background
(music/effects stem + other speakers' stem, different episodes) at several voice-to-background
ratios, optionally with game-like reverb; then
  detection   smallest W at which the strongest match over all lines and all past lags is the right
              line at the right sample, with a margin over every other line
  cancel      residual of the original after subtraction: gain-only (fitted on the first 200 ms) and,
              for reverberant lines, a least-squares FIR fitted on the first 0.5 s
Writes eval/<set>/rt_feasibility.csv and prints a summary.
Usage:  envs/tools/Scripts/python scripts/rt_feasibility.py [--cases 2]
"""
import argparse

import numpy as np
import soundfile as sf
from scipy.linalg import solve_toeplitz
from scipy.signal import fftconvolve

from vl_common import DATA, EVAL, load_lines, read_mono, write_csv

SR = 16000
W_MS = [20, 40, 60, 80, 100, 150, 200, 300]


def onset(x, thr_db=-40):
    e = np.convolve(x ** 2, np.ones(160) / 160, mode="same")
    on = np.flatnonzero(10 * np.log10(e + 1e-12) > 10 * np.log10(e.max() + 1e-12) + thr_db)
    return int(on[0]) if len(on) else 0


def ncc_all(mix, prefixes, norms):
    """Normalised cross-correlation of every prefix (rows, equal length W) at every lag of mix."""
    w = prefixes.shape[1]
    n = len(mix) - w + 1
    nfft = 1 << int(np.ceil(np.log2(len(mix) + w)))
    num = np.fft.irfft(np.fft.rfft(mix, nfft)[None, :] * np.conj(np.fft.rfft(prefixes, nfft, axis=1)), nfft, axis=1)[:, :n]
    c = np.concatenate([[0.0], np.cumsum(mix ** 2)])
    den = np.sqrt(np.maximum(c[w:w + n] - c[:n], 1e-12))[None, :] * norms[:, None]
    return num / den


def background(rng, n, inst, voc):
    def clip(src):
        s = rng.integers(0, len(src) - n)
        return src[s:s + n]
    return clip(inst) + 0.5 * clip(voc)


def reverb_ir(rng, t60=0.35, drr_db=6.0):
    """Room tail under a dominant direct path (direct-to-reverberant ratio drr_db), as for a nearby
    speaker in a game scene; the first version had the tail 16 dB above the direct sound."""
    n = int(t60 * SR)
    tail = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    tail[0] = 0.0
    tail *= np.sqrt(10 ** (-drr_db / 10) / np.sum(tail ** 2))
    tail[0] = 1.0
    return tail


def whiten(x):
    """Pre-emphasis + 300 Hz high-pass: music and effects carry most energy low, the voice cues sit higher."""
    from scipy.signal import butter, sosfilt

    y = np.append(x[0], x[1:] - 0.97 * x[:-1])
    return sosfilt(butter(4, 300, "highpass", fs=SR, output="sos"), y).astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=2, help="random placements per line and condition")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    lines = load_lines()
    wavs = [read_mono(ln["wav"], SR)[0] for ln in lines]
    ons = [onset(w) for w in wavs]
    eps = sorted((DATA / "default" / "vocals").glob("*_inst.wav"))
    inst = read_mono(eps[0], SR)[0]
    voc = read_mono(sorted((DATA / "default" / "vocals").glob("*_vocals.wav"))[1], SR)[0]  # another episode
    wh = [whiten(x) for x in wavs]
    pre = {}
    for wm in W_MS:
        w = int(wm * SR / 1000)
        P = np.zeros((len(wavs), w), dtype=np.float32)
        for k, (x, o) in enumerate(zip(wh, ons)):
            seg = x[o:o + w]
            P[k, :len(seg)] = seg
        pre[wm] = (P, np.sqrt(np.sum(P.astype(np.float64) ** 2, axis=1)) + 1e-9)
    # lines that are the same recording under two file ids (e.g. two 「…」) cannot be told apart and
    # need not be: either replacement is right. Group them by the similarity of their first 300 ms.
    P3, n3 = pre[300]
    sim = (P3 @ P3.T) / np.outer(n3, n3)
    same = sim > 0.9
    print(f"near-duplicate groups: {int((same.sum(axis=1) > 1).sum())} lines share a group")

    rows = []
    conds = [("+6dB", 6, False), ("0dB", 0, False), ("-6dB", -6, False), ("0dB+混响", 0, True)]
    for k, (ln, x) in enumerate(zip(lines, wavs)):
        for cname, vbr, rev in conds:
            for _ in range(a.cases):
                y = fftconvolve(x, reverb_ir(rng))[:len(x)] if rev else x.copy()
                lead = int(rng.uniform(1.0, 2.0) * SR)
                n = lead + len(y) + SR // 2
                bg = background(rng, n, inst, voc)
                o = ons[k]
                v = y[o:o + int(0.5 * SR)]
                b = bg[lead + o:lead + o + len(v)]
                g = np.sqrt(np.mean(b ** 2) + 1e-12) / np.sqrt(np.mean(v ** 2) + 1e-12) * 10 ** (vbr / 20)
                mix = bg.copy()
                mix[lead:lead + len(y)] += g * y
                true = lead + o
                det = None
                for wm in W_MS:
                    P, norms = pre[wm]
                    w = P.shape[1]
                    # causal: the last second of background before the line plus the first W ms of it;
                    # the background lags double as the false-alarm test
                    seg = whiten(mix[true - SR - 64:true + w])[64:]
                    r = ncc_all(seg, P, norms)
                    best = np.unravel_index(np.argmax(r), r.shape)
                    others = np.max(r[~same[k]])  # lines outside this line's duplicate group
                    ok = same[k, best[0]] and abs(best[1] - SR) <= 2 and r[best[0], SR] - others >= 0.1
                    if ok:
                        det = (wm, float(r[best[0], SR]), float(others))
                        break
                # cancellation with the true alignment, fitted causally on the start of the line
                line_part = g * y
                fit = slice(o, min(len(x), o + int(0.2 * SR)))
                gh = np.dot(mix[lead + fit.start:lead + fit.stop], x[fit]) / (np.dot(x[fit], x[fit]) + 1e-12)
                resid = mix[lead:lead + len(x)] - gh * x - bg[lead:lead + len(x)]
                depth_gain = 10 * np.log10(np.sum(line_part ** 2) / (np.sum(resid ** 2) + 1e-12))
                depth_fir = ""
                if rev:  # FIR (128 ms) by least squares on the first 0.5 s, applied to the whole line
                    L, fs = 2048, slice(o, min(len(x), o + SR // 2))
                    xr, dr = x[fs], mix[lead + fs.start:lead + fs.stop]
                    rxx = np.correlate(xr, xr, "full")[len(xr) - 1:len(xr) - 1 + L] + 1e-6
                    rxd = np.correlate(dr, xr, "full")[len(xr) - 1:len(xr) - 1 + L]
                    h = solve_toeplitz(rxx, rxd)
                    est = fftconvolve(x, h)[:len(x)]
                    resid2 = mix[lead:lead + len(x)] - est - bg[lead:lead + len(x)]
                    depth_fir = round(10 * np.log10(np.sum(line_part ** 2) / (np.sum(resid2 ** 2) + 1e-12)), 1)
                rows.append({"id": ln["id"], "dur": round(len(x) / SR, 2), "cond": cname,
                             "detect_ms": det[0] if det else "", "ncc": round(det[1], 3) if det else "",
                             "next_best": round(det[2], 3) if det else "",
                             "cancel_gain_db": round(depth_gain, 1), "cancel_fir_db": depth_fir})
        if k % 30 == 0:
            print(f"{k}/{len(lines)} lines")

    write_csv(EVAL / "rt_feasibility.csv", rows)
    for cname, _, rev in conds:
        rs = [r for r in rows if r["cond"] == cname]
        d = np.array([r["detect_ms"] if r["detect_ms"] != "" else np.inf for r in rs], dtype=float)
        pct = " ".join(f"≤{w}ms:{np.mean(d <= w):.0%}" for w in (40, 60, 80, 100, 150, 300))
        cg = np.median([r["cancel_gain_db"] for r in rs])
        cf = np.median([r["cancel_fir_db"] for r in rs]) if rev else float("nan")
        print(f"{cname:9s} n={len(rs)}  识别 {pct}  未识别 {np.mean(np.isinf(d)):.0%}  "
              f"消除(增益) 中位 {cg:.1f} dB" + (f"  消除(FIR) 中位 {cf:.1f} dB" if rev else ""))
