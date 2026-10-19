"""Replace the character's lines in a gameplay recording with the converted versions (tools env).

Works only on the recorded audio (no game files or process):
  0. capture check  OBS recording both "Desktop Audio" and the window's own audio puts every sound in twice,
                    ~10 ms apart: a comb filter over the whole file that no single-copy model can cancel.
                    Found from the autocorrelation and undone (Y = G (1 + r e^{±jωD})  ->  G).
  1. separate       vocals / rest (Mel-Band RoFormer). In combat the effects bury the voice: a line explains
                    1-3 dB of the mix around it but 5-15 dB of the vocal stem.
  2. find           matched filter of every exported line on the vocal stem -> candidates
  3. verify         how much of the vocal stem a filtered copy of the line explains, against the same fit with
                    the line reversed in time (same spectrum and length, wrong waveform). Only the take that
                    was actually played explains several dB; a similar-sounding other take explains ~0.
  4. remove         subtract the fitted copy (per-bin FIR, refitted every 0.3 s to follow the game's level
                    changes) and suppress what is left where the copy predicts energy (post-filter as in echo
                    cancellers). The change is applied to the mix, so everything else stays bit-identical.
  5. insert         the final version through the line's whole-line filter (scene volume, EQ, early room)
Lines whose final version is the original performance are left untouched. Voice left on the vocal stem
that no line explains is listed for listening (other characters, or lines missing from the export).

Writes exp/<set>/recordings/<name>/{game,removed,replaced}.wav, events.csv, candidates.csv, uncovered.csv,
out/recordings/<name>_02.mp4 and the page eval/<set>/recording_<name>.html.
Usage:  envs/tools/Scripts/python scripts/replace_in_recording.py "<recording>.mp4"
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter
from scipy.signal import butter, fftconvolve, find_peaks, istft, resample_poly, sosfilt, stft, welch

from rt_feasibility import onset
from vl_common import CACHE, EVAL, EXP, ROOT, load_lines, read_csv, read_mono, write_csv
from vl_paths import EXP_BASE, VOICE, char_label, page_path, routes, voice_cfg

SR16, SR = 16000, 48000
NFFT, HOP, TAPS = 1024, 256, 6     # 6 frames of 5.3 ms hop: ~37 ms of reverb/EQ per bin
SEG = 0.3                          # s; the filter is refitted per block of this length
TAIL = 0.3                         # s of room tail after a line
MIN_VOICED = 0.03                  # s; shorter clips are not searched
SHORT = 0.12                       # s; attack breaths (ふっ, はっ) under this need a stricter match
TMPL = 1.5                         # s of each line (from its onset) used by the matched filter
BAND = butter(4, [300, 3000], "bandpass", fs=SR16, output="sos")
BAND48 = butter(4, [300, 3000], "bandpass", fs=SR, output="sos")
PF_OVER, PF_FLOOR = 2.0, 0.1       # post-filter over-subtraction and floor (-20 dB)
CUT_DB = 3.0                       # dB a 0.3 s block of a line must explain to count as still playing
SEP_MODEL = "vocals_mel_band_roformer.ckpt"


def feat(x):
    """Matching band 300-3000 Hz: combat effects sit far above the voice from 4 kHz up, music below 300 Hz."""
    return sosfilt(BAND, x).astype(np.float32)


def to16(y):
    """Mono 16 kHz of a 48 kHz (T x C) array."""
    return resample_poly(y.mean(axis=1), 1, 3).astype(np.float32)


def significance(score, n_samples):
    """z of a normalised correlation over ~2*bandwidth*T independent samples: comparable across lengths."""
    return score * np.sqrt(2 * 2700 * n_samples / SR16)


def ncc_series(x, t):
    num = fftconvolve(x, t[::-1], mode="valid")
    c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    w = len(t)
    return num / (np.sqrt(np.maximum(c[w:] - c[:-w], 1e-12)) * (np.linalg.norm(t) + 1e-9))


def frames(x):
    return stft(x, fs=SR, nperseg=NFFT, noverlap=NFFT - HOP, boundary="zeros", padded=True)[2]


def unframes(Z, n):
    """Inverse of frames(), cut or zero-padded to n samples (channels first, if any)."""
    y = istft(Z, fs=SR, nperseg=NFFT, noverlap=NFFT - HOP, boundary=True)[1]
    return np.pad(y, [(0, 0)] * (y.ndim - 1) + [(0, max(0, n - y.shape[-1]))])[..., :n]


def shifted(X, k):
    return np.pad(X, ((0, 0), (k, 0)))[:, :X.shape[1]]


def taps(X):
    """F x T x K: X delayed by 0..TAPS-1 frames."""
    S = np.zeros(X.shape + (TAPS,), X.dtype)
    for k in range(TAPS):
        S[:, k:, k] = X[:, :X.shape[1] - k]
    return S


def fit_filter(Y, X, lam=1e-3):
    """Per-bin FIR over TAPS frames: Y ≈ Σ_k H[:, k] · X shifted by k (least squares, ridge lam).
    Y may carry channels last (F x T x C): the normal equations depend on X only, so they are solved once."""
    S = taps(X)
    Sh = np.conj(S).transpose(0, 2, 1)                                     # F x K x T
    Rm = Sh @ S                                                             # F x K x K
    Yc = Y if Y.ndim == 3 else Y[..., None]
    load = lam * (np.trace(Rm, axis1=1, axis2=2).real[:, None, None] / TAPS + 1e-9)
    H = np.linalg.solve(Rm + load * np.eye(TAPS)[None], Sh @ Yc)           # F x K x C
    return H if Y.ndim == 3 else H[..., 0]


def fit_apply(Y, X, lam=1e-3):
    """fit_filter and the fitted copy apply_filter(H, X), sharing the delayed-X stack: (H, copy)."""
    S = taps(X)
    Sh = np.conj(S).transpose(0, 2, 1)
    Rm = Sh @ S
    Yc = Y if Y.ndim == 3 else Y[..., None]
    load = lam * (np.trace(Rm, axis1=1, axis2=2).real[:, None, None] / TAPS + 1e-9)
    H = np.linalg.solve(Rm + load * np.eye(TAPS)[None], Sh @ Yc)
    return (H, S @ H) if Y.ndim == 3 else (H[..., 0], (S @ H)[..., 0])


def apply_filter(H, X):
    """Σ_k H[:, k] · X shifted by k; H may carry channels last (F x K x C -> F x T x C)."""
    S = taps(X)
    return S @ H if H.ndim == 3 else (S @ H[..., None])[..., 0]


def fit_blocks(Y, X):
    """The filtered copy of X that best matches Y, with the filter refitted per ~SEG block (each block also
    sees the TAPS frames before it). Follows level changes inside a line (the game's dynamics)."""
    T = Y.shape[1]
    nb = max(1, int(round(T * HOP / SR / SEG)))
    edges = np.linspace(0, T, nb + 1).astype(int)
    est = np.zeros_like(Y)
    for i, b in zip(edges[:-1], edges[1:]):
        a = max(0, i - TAPS)
        est[:, i:b] = apply_filter(fit_filter(Y[:, a:b], X[:, a:b]), X[:, a:b])[:, i - a:]
    return est


def active(X):
    """Bins where the line has sound (within 40 dB of its peak), widened by the filter length (room tail)."""
    P = np.abs(X) ** 2
    m = P > P.max() * 1e-4
    for k in range(1, TAPS):
        m[:, k:] |= m[:, :-k]
    m[:4] = False  # below 190 Hz: music and rumble
    return m


def explained_db(Y, est, m):
    e = np.sum(np.abs(Y[m]) ** 2)
    if e < 1e-12:  # nothing there (or the line's sound falls outside the window)
        return 0.0
    return 10 * np.log10(e / (np.sum(np.abs(Y[m] - est[m]) ** 2) + 1e-20))


def verify(vstem, x, s, n):
    """How much of the vocal stem at [s, s+n) the line placed at s explains, over the part that was played,
    and the same for the line reversed (null). The game cuts a line short when another starts or the player
    moves on, so the match is followed block by block and ends after two blocks in a row where the line has
    sound but the stem does not follow it. Returns (explained dB, null dB, played samples)."""
    xs = np.pad(x, (0, max(0, n - len(x))))[:n]
    Y, X = frames(vstem[s:s + n]), frames(xs)
    est, m = fit_blocks(Y, X), active(X)
    T = Y.shape[1]
    edges = np.linspace(0, T, max(1, int(round(T * HOP / SR / SEG))) + 1).astype(int)
    end, bad = 0, 0
    for i, b in zip(edges[:-1], edges[1:]):
        if m[:, i:b].sum() < 20:  # a pause in the line
            continue
        if explained_db(Y[:, i:b], est[:, i:b], m[:, i:b]) >= CUT_DB:
            end, bad = b, 0
        else:
            bad += 1
            if bad >= 2:
                break
    if end == 0:
        return 0.0, 0.0, 0
    stop = min(T, end + TAPS)
    Yp, Xp = Y[:, :stop], frames(xs[:end * HOP][::-1].copy())
    Yr = frames(vstem[s:s + end * HOP])
    null = explained_db(Yr, fit_blocks(Yr, Xp), active(Xp)) if Xp.shape[1] == Yr.shape[1] else 0.0
    return explained_db(Yp, est[:, :stop], m[:, :stop]), max(0.0, null), end * HOP


# --- 0. capture check -------------------------------------------------------------------------------------
def phat_peak(z, lo, hi):
    Z = np.fft.rfft(z * np.hanning(len(z)), 2 * len(z))
    ac = np.fft.irfft(np.abs(Z) ** 2 / (np.abs(Z) + 1e-9) ** 1.6, 2 * len(z))
    ac /= ac[0]
    k = lo + int(np.argmax(ac[lo:hi]))
    a, b, c = ac[k - 1], ac[k], ac[k + 1]
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) / SR, float(b)


def doubled_capture(y):
    """Inter-copy delay D (s) when the same delay stands out in at least 70% of 10 s windows, else None."""
    m = y.mean(axis=1).astype(np.float64)
    W = min(10 * SR, len(m))
    res = [phat_peak(m[i:i + W], int(0.001 * SR), int(0.1 * SR)) for i in range(0, len(m) - W + 1, W)]
    d = np.array([r[0] for r in res])
    h = np.array([r[1] for r in res])
    D = float(np.median(d))
    ok = (np.abs(d - D) < 5e-5) & (h > 0.05)
    return (D, float(np.median(h))) if ok.mean() >= 0.7 else (None, float(np.median(h)))


def comb_ratio(y, D):
    """|r| that flattens the comb: the long-term spectrum's ripple at the comb's peaks vs notches is zero."""
    f, P = welch(y.mean(axis=1), fs=SR, nperseg=1 << 16)
    sel = (f > 200) & (f < 8000)
    ph = np.cos(2 * np.pi * f[sel] * D)
    best = None
    for r in np.arange(0.05, 0.995, 0.005):
        Q = P[sel] / (1 + r * r + 2 * r * ph)
        rip = abs(10 * np.log10(np.mean(Q[ph > 0.95]) / np.mean(Q[ph < -0.95])))
        if best is None or rip < best[0]:
            best = (rip, float(r))
    return best[1]


def undouble(y, D, r, sign):
    """G = Y / (1 + r e^{sign·jωD}), on the whole file (padded so the r^k tail does not wrap)."""
    n = len(y)
    N = 1 << int(np.ceil(np.log2(n + 2 * SR)))
    Y = np.fft.rfft(np.pad(y.astype(np.float64), ((SR, N - n - SR), (0, 0))), axis=0)
    f = np.fft.rfftfreq(N, 1 / SR)
    g = np.fft.irfft(Y / (1 + r * np.exp(sign * 2j * np.pi * f * D))[:, None], N, axis=0)[SR:SR + n]
    return (g * (1 + r)).astype(np.float32)  # (1 + r): the level of the louder copy plus the other


def pick_sign(y, D, r, lines, o16, ons):
    """Which copy is louder is invisible in the spectrum; the lines tell: undo both ways and keep the one in
    which the strongest matches explain more of the mix."""
    mono16 = feat(to16(y))
    top = []
    for k, x in enumerate(o16):
        t = feat(x)[ons[k]:ons[k] + int(TMPL * SR16)]
        if len(t) < 0.5 * SR16:
            continue
        rr = ncc_series(mono16, t)
        p = int(np.argmax(rr))
        top.append((float(rr[p]), k, p - ons[k]))
    top = sorted(top, reverse=True)[:6]
    score = {}
    for sign in (+1, -1):
        g = undouble(y, D, r, sign).mean(axis=1).astype(np.float64)
        tot = 0.0
        for _, k, s16 in top:
            x = read_mono(lines[k]["wav"], SR)[0].astype(np.float64)
            s, n = s16 * 3, len(x) + int(TAIL * SR)
            if s < 0 or s + n > len(g):
                continue
            X, Y = frames(np.pad(x, (0, n - len(x)))), frames(g[s:s + n])
            tot += explained_db(Y, apply_filter(fit_filter(Y, X), X), active(X))
        score[sign] = tot
    return max(score, key=score.get), score


# --- 1. separate ------------------------------------------------------------------------------------------
def separate(wav, out):
    """vocals.wav / inst.wav in out/, reused while newer than the input."""
    v, i = out / "vocals.wav", out / "inst.wav"
    if v.exists() and i.exists() and v.stat().st_mtime > wav.stat().st_mtime:
        return
    from audio_separator.separator import Separator

    out.mkdir(parents=True, exist_ok=True)
    sep = Separator(output_dir=str(out), model_file_dir=str(CACHE / "separator"), output_format="WAV",
                    sample_rate=SR)
    sep.load_model(model_filename=SEP_MODEL)
    for o in sep.separate(str(wav)):
        o = out / Path(o).name
        o.replace(v if "(vocals)" in o.name.lower() else i)


def voice_path(mix, items, n_fft=4096, half=0.01):
    """The game's voice path (an EQ with non-linear phase) as an FIR of +-half s, from lines matched with
    confidence: items = (x as played, file start in the recording, weight). Per line the cross-spectrum ratio
    on the mix, its residual delay against the strongest line removed, weighted average. Waveform matching
    against the files themselves swings with whichever band dominates; through this path it does not
    (stream_sim.py uses it; quiet-scene lines give the cleanest estimate)."""
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    band = (f > 300) & (f < 3000)

    def spec(z):
        return stft(z, fs=SR, nperseg=n_fft, noverlap=n_fft * 3 // 4, boundary=None, padded=False)[2]

    paths = []
    for x, s, w in items:
        xb, mb = sosfilt(BAND48, x), sosfilt(BAND48, mix[max(0, s - 960):s + len(x) + 960])
        r = fftconvolve(mb, xb[::-1], mode="valid")
        s = max(0, s - 960) + int(np.argmax(np.abs(r)))
        X, Y = spec(x), spec(mix[s:s + len(x)])
        if X.shape[1] == 0 or Y.shape[1] != X.shape[1]:
            continue
        paths.append((np.sum(Y * np.conj(X), 1) / (np.sum(np.abs(X) ** 2, 1) + 1e-12), np.sum(np.abs(X) ** 2, 1) * w))
    if not paths:
        return None
    ref = max(paths, key=lambda h: h[1][band].sum())[0]
    num, den = np.zeros(len(f), complex), np.zeros(len(f))
    for H, W in paths:
        ph = np.unwrap(np.angle(H[band] * np.conj(ref[band])))
        tau = np.polyfit(2 * np.pi * f[band], ph, 1, w=np.sqrt(W[band]))[0]
        H = H * np.exp(-1j * 2 * np.pi * f * tau)
        num += H / (np.abs(H[band]).mean() + 1e-12) * W
        den += W
    h = np.fft.fftshift(np.fft.irfft(num / (den + 1e-12), n_fft))
    L = int(half * SR)
    return h[n_fft // 2 - L:n_fft // 2 + L] * np.hanning(2 * L)


def fit_len(a, n):
    return a[:n] if len(a) >= n else np.pad(a, ((0, n - len(a)), (0, 0)))


# --- 4./5. remove and insert -------------------------------------------------------------------------------
def fade_out(y, ms=60):
    y = y.copy()
    f = min(len(y), int(ms * SR / 1000))
    y[len(y) - f:] *= np.linspace(1, 0, f)  # like the game's stop
    return y


def render(mix, voc, plan):
    """Remove each planned line from the mix and insert its replacement. A plan item has k, s (file start in
    the recording), x (the original as played), fin (the replacement as it should sound, or None to leave
    the line alone), cut (the game stopped it early) and optionally start: samples into the line before which
    nothing changes (a real-time system that recognised the line late). Returns removed, replaced, the
    cleaned vocal stem and per item (voice drop dB, linear-left dB) per channel."""
    removed, replaced = mix.astype(np.float64), mix.astype(np.float64)  # two separate copies
    vwork = voc.astype(np.float64)
    stats = []
    for e in plan:
        s, x, fin = e["s"], e["x"], e["fin"]
        n = min(max(len(x), len(fin) if fin is not None else 0) + int(TAIL * SR), len(mix) - s)
        X = frames(np.pad(x, (0, max(0, n - len(x))))[:n])
        m = active(X)
        if e["cut"]:  # judge the removal on the played part only: right after the cut the next line starts
            m[:, len(x) // HOP:] = False
        gate = np.ones(n)
        st = int(e.get("start", 0))
        if st > 0:  # nothing changes before st, then a 30 ms crossfade
            gate[:st] = 0
            r = min(n - st, int(0.03 * SR))
            gate[st:st + r] = np.linspace(0, 1, r)
        drops, lin = [], []
        if fin is not None:
            Fz = frames(np.pad(fin, (0, max(0, n - len(fin))))[:n])
            for ch in range(mix.shape[1]):
                V = frames(vwork[s:s + n, ch])
                est = fit_blocks(V, X)
                Rz = V - est
                G = np.clip(1 - PF_OVER * np.abs(est) ** 2 / (np.abs(Rz) ** 2 + 1e-12), PF_FLOOR, 1.0)
                Cz = uniform_filter(G, size=(3, 5), mode="nearest") * Rz
                delta = unframes(V - Cz, n) * gate
                vwork[s:s + n, ch] -= delta
                removed[s:s + n, ch] -= delta
                H = fit_filter(V, X)
                replaced[s:s + n, ch] += unframes(apply_filter(H, Fz), n) * gate - delta
                drops.append(10 * np.log10(np.sum(np.abs(V[m]) ** 2) / (np.sum(np.abs(Cz[m]) ** 2) + 1e-20)))
                # honest check: a fresh fit of the line on what is left (LS residuals are orthogonal to the
                # fitted copy by construction, so correlation with the line after subtraction proves nothing)
                lb = np.sum(np.abs(apply_filter(H, X)[m]) ** 2)
                la = np.sum(np.abs(apply_filter(fit_filter(Cz, X), X)[m]) ** 2)
                lin.append(10 * np.log10(lb / (la + 1e-20)))
        stats.append((drops, lin))
    return removed, replaced, vwork, stats


# --- uncovered voice --------------------------------------------------------------------------------------
def uncovered(voc, events, lines):
    """Voice segments on the cleaned vocal stem that no event covers, with an ECAPA similarity to the
    character (a hint only: in combat even true lines score 0.1-0.5)."""
    import torch
    from silero_vad import get_speech_timestamps

    from prep_voice import ecapa, load_vad
    v16 = to16(voc)
    segs = get_speech_timestamps(torch.from_numpy(v16), load_vad(), sampling_rate=SR16,
                                 min_speech_duration_ms=150, min_silence_duration_ms=150, speech_pad_ms=30)
    spk = ecapa()

    def emb(w):
        w = np.tile(w, 8000 // len(w) + 1) if len(w) < 8000 else w
        with torch.no_grad():
            e = spk.encode_batch(torch.from_numpy(w.astype(np.float32))[None].cuda()).squeeze().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    cen = np.mean([emb(read_mono(ln["wav"], SR16)[0]) for ln in lines if float(ln["duration"]) > 1.0], axis=0)
    cen /= np.linalg.norm(cen)
    spans = [(e["v0"] / SR, e["v1"] / SR) for e in events]
    rows = []
    for sg in segs:
        a, b = sg["start"] / SR16, sg["end"] / SR16
        if any(s < b - 0.05 and e > a + 0.05 for s, e in spans):
            continue
        rows.append({"time": round(a, 2), "end": round(b, 2), "dur": round(b - a, 2),
                     "similarity": round(float(emb(v16[sg["start"]:sg["end"]]) @ cen), 2)})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="a recording: video (the result is muxed back) or audio (wav/flac)")
    ap.add_argument("--name", default="", help="work folder name (default: the file name)")
    ap.add_argument("--thr", type=float, default=0.25, help="matched-filter score of a candidate")
    ap.add_argument("--min-z", type=float, default=8.0, help="and its length-aware significance")
    ap.add_argument("--pool-db", type=float, default=3.0, help="vocal stem explained by a candidate to keep it")
    ap.add_argument("--min-db", type=float, default=8.0, help="explained by the line, to accept it")
    ap.add_argument("--min-db-long", type=float, default=5.0, help="same, for lines >= 1 s matching at ncc 0.5")
    ap.add_argument("--margin-db", type=float, default=3.0, help="lead over the time-reversed line")
    ap.add_argument("--min-ncc", type=float, default=0.45, help="matched-filter score, with --min-db")
    ap.add_argument("--min-db-short", type=float, default=10.0, help="--min-db for lines under SHORT of sound")
    ap.add_argument("--min-ncc-short", type=float, default=0.55, help="--min-ncc for them")
    ap.add_argument("--final", default="final_48k", help="exp/ group with the replacement lines")
    ap.add_argument("--no-uncovered", action="store_true", help="skip the list of voice no line explains")
    a = ap.parse_args()

    video = Path(a.video)
    name = a.name or video.stem.replace(" ", "_")
    work = EXP_BASE / "recordings" / name
    work.mkdir(parents=True, exist_ok=True)
    game = work / "game.wav"
    if not game.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "2", "-ar", str(SR),
                        "-c:a", "pcm_f32le", str(game)], check=True)
    raw, _ = sf.read(str(game), dtype="float32", always_2d=True)          # T x 2
    print(f"recording {len(raw) / SR:.1f} s")

    lines = load_lines()
    route = routes()
    final_csv = EXP / a.final / "final.csv"
    final_src = {r["id"]: r["source"] for r in read_csv(final_csv)} if final_csv.exists() else {}
    o16 = [read_mono(ln["wav"], SR16)[0] for ln in lines]
    ons = [onset(x) for x in o16]

    # 0. capture check
    D, h = doubled_capture(raw)
    capture = {"doubled": D is not None, "peak": round(h, 3)}
    clean = game
    if D is not None:
        r = comb_ratio(raw, D)
        sign, sc = pick_sign(raw, D, r, lines, o16, ons)
        capture.update(delay_ms=round(D * 1000, 3), ratio=round(r, 3), louder="later" if sign > 0 else "earlier")
        print(f"capture: every sound is in twice, {D * 1000:.3f} ms apart (the {capture['louder']} copy louder, "
              f"other at {r:.3f}): undoing it (line fit {sc[1]:.1f} vs {sc[-1]:.1f} dB)")
        clean = work / "game_single.wav"
        if not clean.exists() or clean.stat().st_mtime < game.stat().st_mtime:
            sf.write(clean, undouble(raw, D, r, sign), SR, subtype="FLOAT")
    else:
        print(f"capture: single copy (autocorrelation peak {h:.3f})")
    mix, _ = sf.read(str(clean), dtype="float32", always_2d=True)

    # 1. separate
    stems = work / ("stems_single" if D is not None else "stems")
    separate(clean, stems)
    voc = fit_len(sf.read(str(stems / "vocals.wav"), dtype="float32", always_2d=True)[0], len(mix))
    v16 = feat(to16(voc))
    vmid = voc.mean(axis=1).astype(np.float64)
    vmid_b = sosfilt(BAND48, vmid)
    mmid = mix.mean(axis=1).astype(np.float64)

    # 2. find
    voiced = [float(np.sum(np.abs(x[o:]) > 0.05 * np.abs(x).max())) / SR16 for x, o in zip(o16, ons)]
    cands, skipped = [], 0
    for k, x in enumerate(o16):
        t = feat(x)[ons[k]:ons[k] + int(TMPL * SR16)]
        if voiced[k] < MIN_VOICED or np.linalg.norm(t) < 1e-4:
            skipped += 1
            continue
        rr = ncc_series(v16, t)
        thr = a.thr if voiced[k] >= SHORT else a.min_ncc_short  # a 0.1 s breath matches noise at 0.3
        for p in find_peaks(rr, height=thr, distance=max(1, len(t) // 2))[0]:
            if significance(rr[p], len(t)) >= a.min_z:
                cands.append((float(rr[p]), k, int(p - ons[k])))
    print(f"{skipped} lines with under {MIN_VOICED} s of sound are not searched for; {len(cands)} candidates")

    # 3. verify
    x48 = {}
    pool, log = [], {}

    def claim(e, ex, nl, played):
        """Set a candidate's scores and the span it claims: the played part plus the room tail. The line counts
        as cut short only if it still has sound after that point (within 25 dB of its loudest 20 ms);
        otherwise only its quiet tail was left and the whole line stands."""
        x = x48[e["k"]]
        env = np.convolve(x ** 2, np.ones(960) / 960, mode="same")
        if played >= len(x) - int(0.5 * SR) or env[played:].max() < env.max() * 10 ** -2.5:
            played = len(x)
        cut = x[:played]
        loud = np.flatnonzero(np.abs(cut) > 0.05 * np.abs(x).max())
        v0, v1 = e["s"] + (loud[0] if len(loud) else 0), e["s"] + (loud[-1] + 1 if len(loud) else played)
        vm = 10 * np.log10(np.sum(vmid[v0:v1] ** 2) / (np.sum(mmid[v0:v1] ** 2) + 1e-20) + 1e-20)
        e.update(ex=ex, null=nl, played=played, n=min(played + int(TAIL * SR), len(vmid) - e["s"]), v0=v0, v1=v1,
                 cut=played < len(x) - int(0.5 * SR), voiced=len(loud) / SR, vm=vm)
    for score, k, s16 in cands:
        if k not in x48:
            x48[k] = read_mono(lines[k]["wav"], SR)[0].astype(np.float64)
        x = x48[k]
        o = ons[k] * 3
        ref = sosfilt(BAND48, x)[o:o + int(TMPL * SR)]
        lo = s16 * 3 + o - 480
        if lo < 0 or lo + len(ref) + 960 > len(vmid):
            continue
        s = lo + int(np.argmax(np.correlate(vmid_b[lo:lo + len(ref) + 960], ref, "valid"))) - o
        n = min(len(x) + int(TAIL * SR), len(vmid) - s)
        if s < 0 or n <= NFFT:
            continue
        ex, nl, played = verify(vmid, x, s, n)
        if not played:
            continue
        e = {"k": k, "s": s, "ncc": score}
        claim(e, ex, nl, played)
        log[(k, s)] = {"time": round(s / SR, 2), "id": lines[k]["id"], "text": lines[k]["ref_text"],
                       "ncc": round(score, 3), "explained_db": round(ex, 1), "null_db": round(nl, 1),
                       "played_s": round(played / SR, 2), "after_claims_db": "", "accepted": False}
        if ex - nl >= a.margin_db and ex >= a.pool_db:
            pool.append(e)

    def strong(e):
        """A played take explains 11-25 dB of a clean stem and matches by waveform (ncc 0.48-0.88); other takes
        of a similar grunt reach 3-7 dB, and when only a 0.3 s piece of one fits by chance its ncc stays near
        0.3. A long line (>= 1 s of sound) may pass lower: in combat, effects leaking into the stem cap what
        even the right line explains."""
        if e["ex"] - e["null"] < a.margin_db or e["vm"] < -25:  # -25: the stem holds next to nothing there
            return False
        if voiced[e["k"]] < SHORT:
            return e["ex"] >= a.min_db_short and e["ncc"] >= a.min_ncc_short
        return ((e["ex"] >= a.min_db and e["ncc"] >= a.min_ncc) or (e["ex"] >= a.min_db - 2 and e["ncc"] >= 0.6) or
                (e["ex"] >= a.min_db_long and e["ncc"] >= 0.5 and e["voiced"] >= 1.0))

    # claim energy one line at a time, best first: take its fitted copy out of a working stem, then re-check
    # the candidates overlapping it on what is left. A grunt that only matched part of a longer line loses
    # its match once that line is gone.
    resid, events = vmid.copy(), []
    while True:
        ok = [e for e in pool if strong(e)]
        if not ok:
            break
        e = max(ok, key=lambda c: c["ex"])
        pool.remove(e)
        events.append(e)
        log[(e["k"], e["s"])]["accepted"] = True
        s, n = e["s"], e["n"]
        X = frames(np.pad(x48[e["k"]], (0, max(0, n - len(x48[e["k"]]))))[:n])
        resid[s:s + n] -= unframes(fit_blocks(frames(resid[s:s + n]), X), n)
        for c in list(pool):
            if c["s"] < s + n and c["s"] + c["n"] > s:
                xk = x48[c["k"]]
                ex, nl, played = verify(resid, xk, c["s"], min(len(xk) + int(TAIL * SR), len(vmid) - c["s"]))
                log[(c["k"], c["s"])]["after_claims_db"] = round(ex, 1)
                if played:
                    claim(c, ex, nl, played)
                else:
                    pool.remove(c)
    write_csv(work / "candidates.csv", sorted(log.values(), key=lambda r: r["time"]))
    events.sort(key=lambda e: e["s"])
    print(f"{len(events)} lines verified ({len(cands)} candidates)")

    # the voice path, for real-time matching (stream_sim.py): from lines explaining >= 15 dB of the stem
    items = [(x48[e["k"]][:e["played"]], e["s"], 10 ** (e["ex"] / 10)) for e in events
             if e["ex"] >= 15 and e["played"] >= SR]
    h = voice_path(mix.mean(axis=1).astype(np.float64), items) if items else None
    if h is not None:
        np.save(work / "voice_path_fir.npy", h)
        print(f"voice path estimated from {len(items)} lines -> voice_path_fir.npy")

    # 4./5. remove and insert
    plan = []
    for e in events:
        ln = lines[e["k"]]
        src = final_src.get(ln["id"], "")
        keep = not src or src.startswith("N_orig")  # no replacement, or the final is the original itself
        fin = None if keep else read_mono(EXP / a.final / f"{ln['id']}.wav", SR)[0].astype(np.float64)
        x = x48[e["k"]]
        if e["cut"]:  # the game cut the line short: so is the replacement (it follows the original's timing)
            x = x[:e["played"]]
            fin = None if fin is None else fade_out(fin[:e["played"]])
        plan.append({"k": e["k"], "s": e["s"], "x": x, "fin": fin, "cut": e["cut"]})
    removed, replaced, vwork, stats = render(mix, voc, plan)
    rows = []
    for e, (drops, lin) in zip(events, stats):
        s, ln = e["s"], lines[e["k"]]
        src = final_src.get(ln["id"], "")
        keep = not src or src.startswith("N_orig")
        seg = slice(e["v0"], max(e["v1"], e["v0"] + 1))
        mixdb = 10 * np.log10(np.sum(vmid[seg] ** 2) / (np.sum(mmid[seg] ** 2) + 1e-20))
        row = {"time": round(s / SR, 2), "id": ln["id"], "src_wav": ln.get("src_wav", ""), "text": ln["ref_text"],
               "ncc": round(e["ncc"], 3), "explained_db": round(e["ex"], 1), "null_db": round(e["null"], 1),
               "voice_vs_mix_db": round(float(mixdb), 1),
               "voice_drop_db": round(float(np.mean(drops)), 1) if drops else "",
               "linear_left_db": round(float(np.mean(lin)), 1) if lin else "",
               "played_s": round(e["played"] / SR, 2), "cut": "是" if e["cut"] else "",
               "route": route.get(ln["id"], ""), "version": "保留原声" if keep else src}
        rows.append(row)
        print(f"{s / SR:7.2f}s {ln['id']} ex={e['ex']:4.1f}dB (null {e['null']:3.1f}) voice/mix={mixdb:5.1f}dB "
              f"drop={row['voice_drop_db']}dB lin={row['linear_left_db']}dB {'kept' if keep else src} "
              f"「{ln['ref_text'][:20]}」")

    unc = [] if a.no_uncovered else uncovered(vwork, events, lines)
    write_csv(work / "uncovered.csv", unc)
    for nm, sig in (("removed", removed), ("replaced", replaced)):
        peak = float(np.abs(sig).max())
        sf.write(work / f"{nm}.wav", sig / peak * 0.99 if peak > 0.99 else sig, SR, subtype="PCM_16")
    sf.write(work / "game16.wav", raw, SR, subtype="PCM_16")  # 16-bit copy of the recording for the page
    write_csv(work / "events.csv", rows)

    out = ROOT / "out" / "recordings"
    out.mkdir(parents=True, exist_ok=True)
    mp4 = out / f"{name}_{VOICE}.mp4"
    if video.suffix.lower() in (".wav", ".flac", ".mp3", ".m4a", ".ogg"):  # audio only: no video to mux
        mp4 = work / "replaced.wav"
    else:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(work / "replaced.wav"),
                        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                        "-shortest", str(mp4)], check=True)
    data = {"name": name, "game": page_path(work / "game16.wav"), "removed": page_path(work / "removed.wav"),
            "replaced": page_path(work / "replaced.wav"), "events": rows, "uncovered": unc, "capture": capture}
    tpl = (Path(__file__).parent / "templates" / "recording.html").read_text(encoding="utf-8")
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / f"recording_{name}.html").write_text(tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False))
                                                 .replace("__VOICE__", voice_cfg()["label"])
                                                 .replace("__CHAR__", char_label()),
                                                encoding="utf-8")
    print(f"-> {mp4}\n-> {EVAL / f'recording_{name}.html'}")
