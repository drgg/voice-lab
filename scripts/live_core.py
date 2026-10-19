"""Real-time replacement engine (tools env): the rules of stream_sim.py, run incrementally on a live stream.

Feed the game's audio block by block (48 kHz stereo float32) to Engine.push(); it returns the same audio
delayed by `delay` ms with the character's lines replaced. Everything is decided from audio already received:
  candidates  matched filter of the first 100 / 200 ms of each line's real sound (through the voice path)
  judgement   when `delay` - PROC ms of a candidate have been heard (later checkpoints while undecided):
              onset re-aligned to a fixed point, waveform match + explained energy vs the reversed line,
              not a stretch of a line already playing, not the same audio as a line just accepted
  tracking    an accepted line plays to the end of its file unless the character starts another line or
              its continuation stops matching (CUT_RUN s, or CUT_FAST s when the scene goes quiet)
  rendering   per 20 ms step, for each playing line: a per-bin FIR fitted on the 0.4 s around the step
              (the step plus FIT_AHEAD of lookahead), the fitted copy subtracted with the post-filter, the
              final version inserted through the same FIR; steps overlap-added with 10 ms crossfades.
              The fit leans on the line's previous FIR (HOLD): where the original pauses and the final
              version speaks on, the path is held rather than fitted to room noise. GUARD_DB caps the rest.
Every decision goes to `log` (a callable taking a dict), for live_replace.py's session log.
"""
import heapq
import time
from pathlib import Path

import numpy as np
import scipy.fft as sfft
from scipy.ndimage import uniform_filter
from scipy.signal import fftconvolve, sosfilt, sosfilt_zi

import replace_in_recording as R
from rt_feasibility import onset
from vl_common import EXP, load_lines, read_csv, read_mono
from vl_paths import CHAR, char_paths

SR, SR16 = 48000, 16000
HOP = 960                         # 20 ms processing step
PROC = 20                         # ms: a step plus compute, before a judgement can act
CK = [100, 150, 200, 300, 400, 600, 1000, 1500]  # ms heard at which undecided candidates are re-judged
PREFIXES, PREFIX_THR = (0.1, 0.2), 0.45
# Candidates are searched at 8 kHz (every 6th sample of the 300-3000 Hz band; 16 kHz before 2026-10-18): the matched
# filter over every template was the engine's largest cost (14.8 ms every 40 ms with 570 templates, under game load
# steps reached p95 25-41 ms for a 20 ms budget). The judgement re-aligns at 48 kHz; the candidate scores' coarser
# grid is made up by a parabola through each peak. The two prefix groups run on alternate steps (the 0.2 s one with
# the render), so no step carries both.
DEC = 6
SRC = SR // DEC
PEAK_DIST = int(0.025 * SRC)      # one candidate per line per 25 ms
ALIGN = int(0.03 * SR)            # onset re-alignment search, +-
RULE = (3.0, 0.7, 4.0)            # min explained-null dB, min ncc, min explained dB (on the mix, voice path)
# Early judgements that went wrong in play, and what a judgement needs because of them:
# - a line of 1 s or more judged early: false matches at ncc 0.70-0.85 collapse once more is heard, real ones hold
#   or rise. Below each LONG_STEPS heard ms it needs that ncc (one flat 0.85 below 400 ms started real lines up to
#   0.24 s late; the steps start them ~0.1 s earlier at the same count of false ones).
# - a line whose body (BODY_DB of its loudest) starts BODY_LATE ms or more after its onset (one opens on a soft
#   fricative for 220 ms): judged on the soft lead-in alone, wind and blade sounds matched it (7 false replacements in
#   two sessions). Until BODY_MS of the body is heard it needs BODY_NCC (12 lines of the set start that way).
# Traced replays of four sessions scored on the offline pipeline's lines (2026-10-14), against the flat 0.85: false
# replacements 50 -> 39, real lines started 150 ms+ late 21 -> 13, none lost.
LONG_LINE, LONG_STEPS = 1.0, ((200, 0.85), (300, 0.80), (400, 0.75))
BODY_DB, BODY_LATE, BODY_MS, BODY_NCC = 12.0, 150, 100, 0.92
# A line of a character added with --also (Bank.extra) needs ALSO_NCC: that character may not be in the party at
# all. Replaying a the source character-only session with the second character added (2026-10-16), the source character's 107 accepts stayed identical and
# one the second character line was accepted at ncc 0.71 over the source character's grunt 「ん?」; 12 the second character lines put into a session
# through the voice path matched at 0.997-1.0.
ALSO_NCC = 0.85
OVERRIDE, CONT = 0.2, 0.25        # interrupting a playing line: see Engine._judge
TRACK_W = int(0.2 * SR)
CUT_NCC, CUT_RUN, CUT_FAST, CUT_DROP = 0.15, 0.5, 0.1, 8.0
RESYNC = int(0.06 * SR)           # a playing line that stops matching is first searched for +-60 ms: the stream
RESYNC_NCC = 0.7                  # itself can jump (a game hitch, a capture gap) and the line goes on; a shift
RESYNC_MIN = int(0.002 * SR)      # under 2 ms is no jump (an alias of a quiet stretch): left to the stop logic
# Exact digital silence inside a line is a hole in the stream, not the line ending (the game's sound always has
# some noise under a voice): while HOLE_MIN..HOLE_MAX samples of the tracking window are exactly 0, a playing line
# is neither cut nor confirmed. More than HOLE_MAX is the sound really stopping (then the stop logic runs ~0.1 s
# later than it would have). 2026-10-18 session: 73-79 ms holes every 2 s ended lines at a soft syllable.
HOLE_MIN, HOLE_MAX = int(0.0025 * SR), int(0.1 * SR)
RESYNC_TRY, RESYNC_CLEAN = 0.6, 0.85  # also searched when ncc falls under 0.6 after a median of 0.85+ (10 steps)
FIT_BACK, FIT_AHEAD = int(0.25 * SR), int(0.1 * SR)
VERIFY_MAX = int(0.6 * SR)        # the energy test looks at the newest 0.6 s heard (cost bounded)
JUDGE_BUDGET = 0.012              # s of judgements per step; the rest wait one step
HIST = 8                          # s of history kept
XF = int(0.01 * SR)               # crossfade between rendering steps
RHOP = 2 * HOP                    # rendering runs every other step, 40 ms at a time (half the cost)
HOLD = 0.05                       # render fit: ridge toward the line's previous FIR, 5% of its typical energy per
                                  # bin (a silent stretch of the original fitted to room noise gave |H| ~ 800)
GUARD_DB = 12.0                   # inserted version more than 12 dB over the line's path gain predicts: capped
INS_ALPHA = 0.35                  # the FIR that carries the final version in: exponential average over renders
                                  # (time constant ~0.1 s) - refitted every 40 ms it made the voice's colour and
                                  # level waver; taking the original out keeps the fresh fit (exact removal)
MIN_DELAY = (RHOP + XF + FIT_AHEAD) * 1000 // SR + PROC


class Ring:
    """The last `size` samples of a stream, addressed by absolute sample index (zeros where unavailable)."""

    def __init__(self, size, ch=None, dtype=np.float32):
        self.size = size
        self.buf = np.zeros((2 * size,) if ch is None else (2 * size, ch), dtype)
        self.base = self.end = 0

    def append(self, x):
        used, n = self.end - self.base, len(x)
        if n >= self.size:                                    # longer than the history: keep its tail only
            self.buf[:self.size] = x[n - self.size:]
            self.end += n
            self.base = self.end - self.size
            return
        if used + n > len(self.buf):
            drop = used + n - self.size
            self.buf[:used - drop] = self.buf[drop:used]
            self.base += drop
            used -= drop
        self.buf[used:used + n] = x
        self.end += n

    def get(self, a, b):
        out = np.zeros((b - a,) + self.buf.shape[1:], self.buf.dtype)
        lo, hi = max(a, self.base), min(b, self.end)
        if hi > lo:
            out[lo - a:hi - a] = self.buf[lo - self.base:hi - self.base]
        return out


def ncc(seg, t):
    return float(np.dot(seg, t) / (np.linalg.norm(seg) * np.linalg.norm(t) + 1e-9))


def body_onset_ms(x):
    """ms from a line's onset to its body: the first 10 ms frame within BODY_DB of its loudest."""
    n = len(x) // 480
    if n == 0:
        return 0.0
    r = 10 * np.log10((x[:n * 480].astype(np.float64).reshape(n, 480) ** 2).mean(1) + 1e-12)
    return float(np.argmax(r >= r.max() - BODY_DB) * 10)


def line_energy(x):
    """Per-bin energy of a line's sounding frames (frames within 20 dB of its loudest), floored at -40 dB."""
    P = np.abs(R.frames(x.astype(np.float32))) ** 2
    fe = P.sum(0)
    E = P[:, fe > 0.01 * fe.max()].mean(1)
    return np.maximum(E, 1e-4 * E.max()).astype(np.float32)


class Bank:
    """Every exported line: original, matching templates (through the voice path), final version."""

    def __init__(self, final="final_48k", path_fir=None, log=print, also=()):
        """also: more characters replaced in the same session (the user, 2026-10-16: 把第二个角色的a组加入实时替换),
        each "char:set:voice:group" - its lines, finished group and voice path from vl_paths.char_paths. Their ids
        carry the character ("other:L001"); the current character's stay bare, as before."""
        srcs = [{"char": "", "lines": load_lines(), "exp": EXP, "final": final, "fir": path_fir}]
        for spec in also:
            char, set_name, voice, group = spec.split(":")
            p = char_paths(char, set_name, voice)
            if not (p["exp"] / group / "final.csv").exists():
                log({"type": "bank_skip", "also": spec, "missing": str(p["exp"] / group / "final.csv")})
                continue
            srcs.append({"char": char, "lines": load_lines(p["lines"]), "exp": p["exp"], "final": group,
                         "fir": p["voice_path"]})
        lines, fin_path, keep, firs, self.extra, self.char = [], [], [], [], [], []
        for s in srcs:
            fs = {r["id"]: r["source"] for r in read_csv(s["exp"] / s["final"] / "final.csv")}
            h = np.load(s["fir"]) if s["fir"] is not None and Path(s["fir"]).exists() else None
            for ln in s["lines"]:
                lines.append({**ln, "id": f"{s['char']}:{ln['id']}" if s["char"] else ln["id"]})
                self.extra.append(bool(s["char"]))
                self.char.append(s["char"] or CHAR)
                keep.append(fs.get(ln["id"], "N_orig").startswith("N_orig"))
                fin_path.append(s["exp"] / s["final"] / f"{ln['id']}.wav")
                firs.append(h)
        self.chars = list(dict.fromkeys(self.char))  # the characters, current one first (Engine.set_chars)
        self.lines, self.ids = lines, [ln["id"] for ln in lines]
        # single precision throughout (the rendering already works in it): half the memory, which the game
        # needs - under its load the engine's pages were swapped out and steps stalled for 300 ms
        self.x48 = [read_mono(ln["wav"], SR)[0].astype(np.float32) for ln in lines]
        self.on = [onset(read_mono(ln["wav"], SR16)[0]) * 3 for ln in lines]           # 48 kHz
        log({"type": "bank", "lines": len(lines), "voice_path": str(path_fir) if firs and firs[0] is not None else None,
             "sources": [{"char": s["char"] or "(current)", "final": str(s["exp"] / s["final"]), "lines": len(s["lines"])}
                         for s in srcs]})
        xm = [fftconvolve(x, h)[len(h) // 2:len(h) // 2 + len(x)] if h is not None else x for x, h in zip(self.x48, firs)]
        self.b48 = [sosfilt(R.BAND48, x).astype(np.float32) for x in xm]  # waveform-matching templates, the stream's own chain
        self.f16 = [b[::3] for b in self.b48]             # ... at 16 kHz (band < 3 kHz), for SIM below
        loud = [np.flatnonzero(np.abs(x) > 0.05 * np.abs(x).max()) for x in self.x48]
        self.voiced_line = [(lo[-1] - self.on[k]) / SR for k, lo in enumerate(loud)]
        self.short = [len(lo) / SR < R.SHORT for lo in loud]
        self.body_on = [body_onset_ms(x[o:]) for x, o in zip(self.x48, self.on)]
        self.keep = keep
        self.fin = [None if self.keep[k] else read_mono(fin_path[k], SR)[0].astype(np.float32)
                    for k in range(len(self.ids))]
        # per-bin energy of each replaced line's sounding frames (the render fit's floor), computed here: done at
        # a line's first render it was a whole-line STFT inside one 20 ms step (spikes of 100+ ms)
        self.E = [None if self.keep[k] else line_energy(x) for k, x in enumerate(self.x48)]
        # candidate templates, grouped by length
        self.groups = []
        fc = [b[::DEC] for b in self.b48]                # candidate rate (SRC), the stream's own decimation
        for pre in PREFIXES:
            n = int(pre * SRC)
            rows, ks, st = [], [], []
            for k in range(len(lines)):
                if len(loud[k]) < R.MIN_VOICED * SR:
                    continue
                sc = max(self.on[k], loud[k][0]) // DEC
                t = fc[k][sc:sc + n]
                if len(t) < n or np.linalg.norm(t) < 1e-4:
                    continue
                rows.append(t)
                ks.append(k)
                st.append(sc)
            T = np.stack(rows)
            self.groups.append({"n": n, "pre": pre, "T": T, "norm": np.linalg.norm(T, axis=1), "k": np.array(ks),
                                "stc": np.array(st)})
        # same audio in two files (lag-tolerant, 300 ms from the onset)
        P = np.stack([np.pad(f[o // 3:o // 3 + 4800], (0, max(0, 4800 - len(f[o // 3:o // 3 + 4800]))))
                      for f, o in zip(self.f16, self.on)])
        nP = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        G = int(0.02 * SR16)
        nfft = 1 << int(np.ceil(np.log2(P.shape[1] + 2 * G)))
        FP, FQ = np.fft.rfft(np.pad(nP, ((0, 0), (G, G))), nfft, axis=1), np.conj(np.fft.rfft(nP, nfft, axis=1))
        S = np.stack([np.abs(np.fft.irfft(FP[i][None] * FQ, nfft, axis=1)[:, :2 * G + 1]).max(axis=1)
                      for i in range(len(nP))])
        self.SIM = np.maximum(S, S.T)


class Engine:
    def __init__(self, bank, delay_ms=200, rule=RULE, log=print):
        if delay_ms < MIN_DELAY:
            raise ValueError(f"delay must be at least {MIN_DELAY} ms")
        self.b, self.log, self.rule = bank, log, rule
        self.D = int(delay_ms * SR / 1000)
        # ms heard at the first judgement: early enough that the line's start is not rendered yet (renders
        # run 40 ms at a time, so up to one more step ahead)
        self.first = delay_ms - PROC - (RHOP - HOP) * 1000 // SR
        self.raw = Ring(HIST * SR, 2)
        self.mono = Ring(HIST * SR, None, np.float64)
        self.mb = Ring(HIST * SR, None, np.float64)
        self.fc = Ring(HIST * SRC, None, np.float64)       # the band at the candidate rate
        self.acc = Ring(HIST * SR, 2, np.float64)         # overlap-added change: insertion minus removal
        self.zi = sosfilt_zi(R.BAND48) * 0.0
        self.t_in = self.t_proc = 0
        self.donec = [0 for _ in bank.groups]
        self.groups, self.enabled = bank.groups, set(bank.chars)   # every character's lines matched (set_chars)
        self.best = [(np.full(len(g["k"]), -1.0), np.zeros(len(g["k"]), int)) for g in bank.groups]
        self.last_cand = {}
        self.cands, self.queue, self.active = [], [], []
        self.bypass = False
        self.cpu = []
        self.errors = {}
        self.r_end = 0                                    # output rendered (final but for crossfades) up to here

    # --- input -------------------------------------------------------------------------------------------
    def push(self, block, fast=False):
        """block: (n, 2) float32. Returns the output that became final: (m, 2) float32, delayed by `delay`.
        fast=True only buffers and passes audio through (catching up after an overload)."""
        self.raw.append(block.astype(np.float32))
        m = block.mean(axis=1).astype(np.float64)
        self.mono.append(m)
        mb, self.zi = sosfilt(R.BAND48, m, zi=self.zi)
        self.mb.append(mb)
        first = (-self.t_in) % DEC                        # keep every sample whose absolute index is 0 mod DEC
        self.fc.append(mb[first::DEC])
        self.acc.append(np.zeros((len(block), 2)))       # the change follows the input timeline
        self.t_in += len(block)
        out = []
        while self.t_in - self.t_proc >= HOP:
            self.t_proc += HOP
            t0 = time.perf_counter()
            if fast:
                out.append(self.raw.get(self.t_proc - self.D, self.t_proc - self.D + HOP))
                continue
            self._step(self.t_proc)
            out.append(self._emit(self.t_proc))
            self.cpu.append(time.perf_counter() - t0)
        return np.concatenate(out) if out else np.zeros((0, 2), np.float32)

    # --- one 20 ms step ----------------------------------------------------------------------------------
    def _step(self, T):
        """Each stage on its own: a bug in one is logged and costs only that stage (never the game's audio)."""
        # candidates: each prefix group every 40 ms (all positions covered), the groups on alternate steps
        self._guard(self._candidates, (T // HOP) % 2, T)
        t0 = time.perf_counter()
        while self.queue and self.queue[0][0] <= T and time.perf_counter() - t0 < JUDGE_BUDGET:
            _, c, heard = heapq.heappop(self.queue)
            self._guard(self._judge, c, heard, T)
        self._guard(self._track, T)
        if (T // HOP) % 2 == 0:
            self._guard(self._render, T)
            self.r_end = T - self.D + RHOP

    def _guard(self, fn, *args):
        try:
            fn(*args)
        except Exception as e:
            import traceback
            key = (fn.__name__, repr(e)[:80])
            self.errors[key] = self.errors.get(key, 0) + 1
            if self.errors[key] <= 3 or self.errors[key] % 100 == 0:  # the same error once per 100 after 3
                self.log({"type": "exception", "t": args[-1] / SR, "stage": fn.__name__, "error": repr(e),
                          "count": self.errors[key], "trace": traceback.format_exc(limit=6)})

    def set_chars(self, enabled):
        """Match only these characters' lines (Bank.chars names), e.g. the ones in the party (the user,
        2026-10-17: 根据角色是否在当前队伍进行语音的取舍): only their templates go into the per-step correlation,
        which is most of the matching cost. A line already playing plays on; candidates of a character switched
        off and still waiting for their judgement are dropped."""
        enabled = set(enabled) & set(self.b.chars)
        if enabled == self.enabled:
            return
        self.enabled = enabled
        self.groups = []
        for g in self.b.groups:
            m = np.array([self.b.char[k] in enabled for k in g["k"]], bool)
            self.groups.append({"n": g["n"], "pre": g["pre"], "T": g["T"][m], "norm": g["norm"][m], "k": g["k"][m],
                                "stc": g["stc"][m]})
        self.best = [(np.full(len(g["k"]), -1.0), np.zeros(len(g["k"]), int)) for g in self.groups]
        for c in self.cands:
            if not c["done"] and self.b.char[c["k"]] not in enabled:
                c["done"] = True
        self.log({"type": "chars", "t": round(self.t_in / SR, 2), "on": sorted(enabled),
                  "templates": int(sum(len(g["k"]) for g in self.groups))})

    def _candidates(self, phase, T):
        """The groups whose turn it is (group i on steps of phase (i + 1) % 2: the 0.1 s group where the
        candidates always ran, the 0.2 s group with the render): matched filter over the positions completed
        since the group's last turn."""
        Tc = T // DEC
        for gi, g in enumerate(self.groups):
            if (gi + 1) % 2 != phase:
                continue
            if not len(g["k"]):
                self.donec[gi] = max(self.donec[gi], Tc - g["n"] + 1)
                continue
            n = g["n"]
            p0, p1 = self.donec[gi], Tc - n + 1              # template start positions now complete
            if p1 <= p0:
                continue
            seg = self.fc.get(p0, p1 + n - 1)
            nfft = 1 << int(np.ceil(np.log2(len(seg))))
            FT = g.setdefault("fft", {}).get(nfft)
            if FT is None:                                    # template spectra (and a product buffer), once
                FT = g["fft"][nfft] = np.conj(sfft.rfft(g["T"].astype(np.float32), nfft, axis=1))
                g.setdefault("buf", {})[nfft] = np.empty_like(FT)
            buf = g["buf"][nfft]
            np.multiply(sfft.rfft(seg.astype(np.float32), nfft)[None], FT, out=buf)
            num = sfft.irfft(buf, nfft, axis=1, overwrite_x=True, workers=-1)[:, :p1 - p0]
            c = np.concatenate([[0.0], np.cumsum(seg ** 2)])
            e = np.sqrt(np.maximum(c[n:n + p1 - p0] - c[:p1 - p0], 1e-12))
            S = num / (e[None] * g["norm"][:, None] + 1e-9)
            bs, bp = self.best[gi]                            # pending peak per template (one per 25 ms)
            j = S.argmax(axis=1)
            cm = S[np.arange(len(S)), j]
            r = np.flatnonzero((j > 0) & (j < S.shape[1] - 1))   # the peak between grid points (parabola)
            if len(r):
                y0, y1, y2 = S[r, j[r] - 1], S[r, j[r]], S[r, j[r] + 1]
                den = y0 - 2 * y1 + y2
                ok = den < 0
                cm[r[ok]] = (y1 - (y0 - y2) ** 2 / (8 * np.where(ok, den, -1.0)))[ok]
            cj = j + p0
            far = cj - bp >= PEAK_DIST                        # this step's best is a new peak: settle the old
            for i in np.flatnonzero(far & (bs >= PREFIX_THR)):
                self._new_candidate(int(g["k"][i]), int(bp[i]), int(g["stc"][i]), g["pre"])
            bs[far] = -1.0
            up = cm > bs
            bs[up], bp[up] = cm[up], cj[up]
            ripe = (p1 - bp >= PEAK_DIST) & (bs >= PREFIX_THR)  # nothing higher can come within 25 ms any more
            for i in np.flatnonzero(ripe):
                self._new_candidate(int(g["k"][i]), int(bp[i]), int(g["stc"][i]), g["pre"])
            bs[ripe] = -1.0
            self.donec[gi] = p1

    def _new_candidate(self, k, pc, stc, pre):
        b = self.b
        o48 = (pc - stc) * DEC + b.on[k]
        last = self.last_cand.get(k)
        if last is not None and abs(o48 - last) < PEAK_DIST * DEC:
            return                                            # already a candidate (other prefix length)
        self.last_cand[k] = o48
        avail = (stc * DEC - b.on[k]) / SR * 1000 + pre * 1000
        full = max(b.voiced_line[k] * 1000, 1.0)
        heard = max(min(self.first, full), avail)
        self.cands.append({"k": k, "o48": o48, "done": False})
        heapq.heappush(self.queue, (o48 + int(heard * SR / 1000), len(self.cands) - 1, heard))

    # --- judgement -----------------------------------------------------------------------------------------
    def _evidence(self, k, o48, heard):
        b = self.b
        n = min(b.on[k] + int(heard * SR / 1000), len(b.x48[k]))
        t = b.b48[k][b.on[k]:n]
        nc = -1.0
        for _ in range(4):                                    # re-centre until the best match stops moving
            lo = o48 - ALIGN
            if lo < b.on[k]:
                return None
            seg = self.mb.get(lo, lo + len(t) + 2 * ALIGN)
            r = R.ncc_series(seg, t)
            i = int(np.argmax(r))
            moved = abs(lo + i - o48) > 2
            o48, nc = lo + i, float(r[i])
            if not moved:
                break
        else:
            return nc, 0.0, 0.0, 0, o48
        if nc < self.rule[1] - 0.1:                           # cannot pass now; maybe at a later checkpoint
            return nc, 0.0, 0.0, 0, o48
        s = o48 - b.on[k]
        i0 = max(b.on[k], n - VERIFY_MAX)                     # energy test on the newest 0.6 s heard
        sig = self.mono.get(s + i0, s + n)
        ex, nl, played = R.verify(sig, b.x48[k][i0:n], 0, n - i0)
        return nc, ex, nl, (i0 + played) if played else 0, o48

    def _continuation(self, o48, heard):
        """Best match, over what the candidate heard, of a playing line, and how well that line matched
        when accepted."""
        b, n = self.b, int(heard * SR / 1000)
        best = (-1.0, 0.0)
        for a in self.active:
            end = a["end"] if a["end"] is not None else a["s"] + len(b.x48[a["k"]])
            if not a["o48"] < o48 < end:
                continue
            i = b.on[a["k"]] + (o48 - a["o48"])
            t = b.b48[a["k"]][i:i + n]
            if len(t) < n // 2 or np.linalg.norm(t) < 0.1 * np.linalg.norm(b.b48[a["k"]][b.on[a["k"]]:][:n]):
                continue
            best = max(best, (ncc(self.mb.get(o48, o48 + len(t)), t), a["ncc"]))
        return best

    def _rules(self, k, ex, nl, nc, heard=1e9):
        dn, nmin, emin = self.rule
        nmin = nmin + (0.05 if self.b.short[k] else 0.0)
        if self.b.voiced_line[k] >= LONG_LINE:
            nmin = max(nmin, next((m for h, m in LONG_STEPS if heard < h), 0.0))
        if self.b.body_on[k] >= BODY_LATE and heard < self.b.body_on[k] + BODY_MS:
            nmin = max(nmin, BODY_NCC)
        if self.b.extra[k]:
            nmin = max(nmin, ALSO_NCC)
        return ex - nl >= dn and nc >= nmin and ex >= emin

    def _judge(self, c, heard, T):
        cand, b = self.cands[c], self.b
        if cand["done"]:
            return
        k = cand["k"]
        cont, cont_acc = self._continuation(cand["o48"], heard)
        if cont >= 0.8:                                       # inside a line still matching this well: no
            cand["done"] = True                               # candidate could interrupt it (ncc <= 1)
            return
        ev = self._evidence(k, cand["o48"], heard)
        if ev is None:
            cand["done"] = True
            return
        nc, ex, nl, played, o48 = ev
        cand["o48"] = o48
        if any(abs(o48 - a["o48"]) < 0.03 * SR and b.SIM[k, a["k"]] >= 0.9 for a in self.active + self.recent):
            cand["done"] = True
            return
        cont, cont_acc = self._continuation(o48, heard)
        if cont >= CONT and (nc < cont + OVERRIDE or nc < cont_acc + 0.05):
            cand["done"] = True                               # a stretch of the line already playing
            return
        if played and self._rules(k, ex, nl, nc, heard):
            cand["done"] = True
            self._accept(k, o48, heard, nc, ex, nl, T)
            return
        nxt = [h for h in CK if h > heard]
        full = b.voiced_line[k] * 1000
        if heard >= full or not nxt or (heard >= 150 and nc < min(0.3, self.rule[1] - 0.05)):
            cand["done"] = True
            if nc >= 0.6 or ex >= 6:                          # close calls: what a later review looks at
                self.log({"type": "reject", "t": o48 / SR, "id": b.ids[k], "heard_ms": round(heard),
                          "ncc": round(nc, 3), "ex": round(ex, 1), "null": round(nl, 1)})
            return
        h = min(nxt[0], full) if full > heard else nxt[0]
        heapq.heappush(self.queue, (o48 + int(h * SR / 1000), c, h))

    @property
    def recent(self):
        return getattr(self, "_recent", [])

    def _accept(self, k, o48, heard, nc, ex, nl, T):
        b = self.b
        s = o48 - b.on[k]
        D = heard + PROC
        emitted = max(T - self.D, self.r_end)                 # output is rendered up to here
        start = max(0, emitted - s) if emitted > o48 else 0   # recognised late: change only what is still ahead
        # s: where the original line starts in the stream (moved by a resync); fs: where the final version
        # starts, kept - after a jump in the stream it plays on without a jump of its own
        a = {"k": k, "o48": o48, "s": s, "fs": s, "ncc": nc, "end": None, "cut": False, "start": start, "run": 0.0,
             "run_at": None, "level": [], "keep": b.keep[k], "E": None, "H": None, "g2": None, "guard": 0}
        # one line at a time per character: the newest ends that character's others. Another character's line
        # plays on (the game plays both; 2026-10-18: the source character's 「…」 matched in a pause of the second character's line and cut
        # it 5 s early) - it stops when tracking loses it
        for p in self.active:
            if b.char[p["k"]] != b.char[k]:
                continue
            if p["o48"] < o48 and (p["end"] is None or p["end"] > o48):
                p["end"], p["cut"] = max(o48, p["s"]), True
                over = max(0, emitted - o48) / SR * 1000
                self.log({"type": "cut", "t": o48 / SR, "id": b.ids[p["k"]], "by": b.ids[k],
                          "overrun_ms": round(over)})
        self.active.append(a)
        self._recent = (self.recent + [a])[-8:]
        self.log({"type": "accept", "t": o48 / SR, "id": b.ids[k], "text": b.lines[k]["ref_text"][:40],
                  "decided_ms": round(D), "late_ms": round(max(0, (emitted - o48) / SR * 1000)),
                  "ncc": round(nc, 3), "ex": round(ex, 1), "null": round(nl, 1), "replace": not b.keep[k]})

    # --- tracking -----------------------------------------------------------------------------------------
    def _track(self, T):
        b = self.b
        for a in self.active:
            if a["end"] is not None:
                continue
            k, s = a["k"], a["s"]
            i = T - TRACK_W - s                               # template index of the newest full window
            if i + TRACK_W // 2 >= len(b.x48[k]):
                a["end"] = s + len(b.x48[k])
                continue
            if i < b.on[k]:
                continue
            z = int(np.count_nonzero(self.mono.get(T - TRACK_W, T) == 0))
            if HOLE_MIN <= z <= HOLE_MAX:                     # a hole in the stream (a capture or game hitch):
                a["hole"] = True                              # no evidence either way until it has passed (also
                continue                                      # in a pause of the line: searched when it speaks)
            t = b.b48[k][i:i + TRACK_W]                       # (shorter at the end of the file)
            if np.linalg.norm(t) < 0.05 * np.linalg.norm(b.b48[k][b.on[k]:b.on[k] + TRACK_W]):
                continue                                      # a pause in the line
            seg = self.mb.get(T - TRACK_W, T - TRACK_W + len(t))
            c = ncc(seg, t)
            cs = a.setdefault("cs", [])
            # a jump is looked for when the line is lost, right after a hole, and when a line that matched cleanly
            # suddenly does not: a 30 ms shift can leave ncc 0.2-0.6 - not lost, but the removal no longer fits
            # (the original leaks) and a false candidate can interrupt the line (2026-10-18: a 90 ms hole, then
            # the source character's 「…」 cut the second character's line 5 s before its end)
            jumped = a.pop("hole", False) or (c < RESYNC_TRY and len(cs) >= 5 and np.median(cs) >= RESYNC_CLEAN)
            if c < CUT_NCC or jumped:                         # lost it: the stream may have jumped
                # the newest window searched along the line +-RESYNC, so both ways are found: audio lost (the
                # line further on) and audio inserted (further back). The stream side could only find the first:
                # its later side is not heard yet (2026-10-18: 30 ms inserted every 2 s lost every the second character line)
                lo = max(b.on[k], i - RESYNC)
                wide = b.b48[k][lo:i + len(t) + RESYNC]
                if len(wide) > len(t):
                    r = R.ncc_series(wide, seg)
                    j = int(np.argmax(r))
                    shift = i - (lo + j)                      # > 0: the line now comes that much later
                    if r[j] >= max(RESYNC_NCC, c + 0.1) and abs(shift) >= RESYNC_MIN:
                        a["s"] += shift
                        a["o48"] += shift
                        t, c = wide[j:j + len(seg)], float(r[j])
                        self.log({"type": "resync", "t": round(T / SR, 2), "id": b.ids[k],
                                  "shift_ms": round(shift / SR * 1000, 1), "ncc": round(c, 2)})
            rms, trms = float(np.sqrt(np.mean(seg ** 2)) + 1e-12), float(np.sqrt(np.mean(t ** 2)) + 1e-12)
            if c < CUT_NCC:
                a["run_at"] = T - TRACK_W if a["run_at"] is None else a["run_at"]
                a["run"] += HOP / SR
                quiet = bool(a["level"]) and bool(20 * np.log10(np.median(a["level"]) / (rms / trms)) >= CUT_DROP)
                if a["run"] >= CUT_RUN or (quiet and a["run"] >= CUT_FAST):
                    a["end"], a["cut"] = max(a["run_at"], s), True
                    over = max(0, (T - self.D) - a["end"]) / SR * 1000
                    self.log({"type": "stop", "t": a["end"] / SR, "id": b.ids[k], "quiet": quiet,
                              "overrun_ms": round(over)})
            else:
                a["run"], a["run_at"] = 0.0, None
                a["level"].append(rms / trms)
                a["level"] = a["level"][-200:]
                cs.append(c)
                del cs[:-10]
        # forget lines whose rendering is over
        self.active = [a for a in self.active if self._extent(a) > T - self.D - FIT_BACK]

    def _extent(self, a):
        """End of what a line changes: its cut point (plus the filter's reach), else the longer of original
        and final version (plus the same)."""
        x, fin = self.b.x48[a["k"]], self.b.fin[a["k"]]
        reach = int(0.06 * SR)                                # 6 taps of 5.3 ms, rounded up
        if a["cut"]:
            return a["end"] + reach
        return max(a["s"] + len(x), a.get("fs", a["s"]) + (len(fin) if fin is not None else 0)) + reach

    # --- rendering ----------------------------------------------------------------------------------------
    def _fit(self, a, V, X):
        """R.fit_apply with the ridge pulled toward the line's previous FIR and floored at HOLD of the line's
        typical energy per bin: (H, fitted copy). A window where the original is silent keeps the last FIR."""
        if a["E"] is None:                                    # per-bin energy of the line's sounding frames
            E = self.b.E[a["k"]] if getattr(self.b, "E", None) else None
            a["E"] = E if E is not None else line_energy(self.b.x48[a["k"]])
        S = R.taps(X)
        Sh = np.conj(S).transpose(0, 2, 1)
        Rm = Sh @ S
        tr = np.trace(Rm, axis1=1, axis2=2).real / R.TAPS
        load = (1e-3 * tr + HOLD * X.shape[1] * a["E"] + 1e-12).astype(np.float32)[:, None, None]
        rhs = Sh @ V if a["H"] is None else Sh @ V + load * a["H"]
        # single precision all the way: a float64 identity here once turned the solve, the filter, the post-filter
        # and the inverse transform into double precision (about twice the time per render)
        H = np.linalg.solve(Rm + load * np.eye(R.TAPS, dtype=np.float32)[None], rhs)
        a["H"] = H
        est = S @ H
        ex = float(tr.sum())
        if ex > 0.3 * X.shape[1] * float(a["E"].sum()):      # the original sounds here: its path gain
            g2 = float(np.sum(np.abs(est) ** 2)) / V.shape[-1] / ex
            a["g2"] = g2 if a["g2"] is None else 0.8 * a["g2"] + 0.2 * g2
        return H, est

    def _cap(self, a, ins, Fz, t):
        """Caps the inserted version at GUARD_DB over what the line's path gain predicts (logged)."""
        if a["g2"] is None:
            return ins
        e = float(np.sum(np.abs(ins) ** 2)) / ins.shape[-1]
        lim = a["g2"] * float(np.sum(np.abs(Fz) ** 2)) * 10 ** (GUARD_DB / 10)
        if e <= lim or e < 1e-12:
            return ins
        a["guard"] += 1
        if a["guard"] in (1, 10, 100):
            self.log({"type": "render_guard", "t": t / SR, "id": self.b.ids[a["k"]], "over_db":
                      round(10 * np.log10(e / lim) + GUARD_DB, 1), "count": a["guard"]})
        return ins * np.float32(np.sqrt(lim / e))

    def _render(self, T):
        b = self.b
        tau = T - self.D + XF                                 # this render's output interval: [tau, tau + RHOP)
        a0, a1 = tau - XF, tau + RHOP + XF
        w0, w1 = tau - FIT_BACK, tau + RHOP + FIT_AHEAD
        win = np.ones(a1 - a0)
        win[:2 * XF] = np.linspace(0, 1, 2 * XF)
        win[-2 * XF:] = np.linspace(1, 0, 2 * XF)
        for a in self.active:
            if a["keep"]:
                continue
            k, s = a["k"], a["s"]
            x, fin = b.x48[k], b.fin[k]
            if a1 <= s + a["start"] or a0 >= self._extent(a):
                continue
            fs = a.get("fs", s)
            xe = min(a["end"], s + len(x)) - s if a["cut"] else len(x)     # original as played (relative)
            fe = a["end"] - fs if a["cut"] else len(fin)                     # replacement, cut like it
            X = R.frames(self._slice(x, xe, w0 - s, w1 - s).astype(np.float32))  # single precision: enough,
            fz = self._slice(fin, min(fe, len(fin)), w0 - fs, w1 - fs, fade_at=fe - (w0 - fs) if a["cut"] else None)
            Fz = R.frames(fz.astype(np.float32))                                 # and twice as fast
            V2 = self.raw.get(w0, w1)
            gate = np.ones(a1 - a0)
            st = s + a["start"]
            if st > a0:                                       # recognised late: crossfade in from here
                gate[:max(0, st - a0)] = 0
                r0 = max(0, st - a0)
                r = min(len(gate) - r0, int(0.03 * SR))
                gate[r0:r0 + r] = np.linspace(0, 1, r)
            n = w1 - w0
            V = np.moveaxis(R.frames(V2.T), 0, -1)            # F x T x 2: both channels share the fit's X side
            H, est = self._fit(a, V, X)
            Rz = V - est
            G = np.clip(1 - R.PF_OVER * np.abs(est) ** 2 / (np.abs(Rz) ** 2 + 1e-12), R.PF_FLOOR, 1.0)
            Cz = uniform_filter(G, size=(3, 5, 1), mode="nearest") * Rz
            # the change, in one inverse transform: the final version inserted, the voice (V - Cz) taken out
            a["Hs"] = H if a.get("Hs") is None else (1 - INS_ALPHA) * a["Hs"] + INS_ALPHA * H
            ins = self._cap(a, R.apply_filter(a["Hs"], Fz), Fz, tau)
            chg = R.unframes(np.moveaxis(ins - (V - Cz), -1, 0), n)                     # 2 x n
            g = win * gate
            for ch in range(2):
                self._add(self.acc, a0, a1, ch, chg[ch, a0 - w0:a1 - w0] * g)

    @staticmethod
    def _slice(y, valid, a, b, fade_at=None):
        """y[a:b] with zeros outside [0, valid); a 60 ms fade ending at `fade_at` (a cut replacement)."""
        out = np.zeros(b - a)
        lo, hi = max(a, 0), min(b, valid, len(y))
        if hi > lo:
            out[lo - a:hi - a] = y[lo:hi]
        if fade_at is not None:
            f = int(0.06 * SR)
            i1 = int(fade_at)
            i0 = max(0, i1 - f)
            if i1 > 0 and i0 < len(out):
                seg = np.linspace(1, 0, i1 - i0)[:max(0, min(len(out), i1) - i0)]
                out[i0:i0 + len(seg)] *= seg
        return out

    def _add(self, ring, a0, a1, ch, y):
        # accumulators run on the input's timeline (rendering never reaches past the newest input)
        lo, hi = max(a0, ring.base), min(a1, ring.end)
        if hi > lo:
            ring.buf[lo - ring.base:hi - ring.base, ch] += y[lo - a0:hi - a0]

    def _emit(self, T):
        a, b_ = T - self.D, T - self.D + HOP
        out = self.raw.get(a, b_).astype(np.float64)
        if not self.bypass:
            out = out + self.acc.get(a, b_)
        return np.clip(out, -1.0, 1.0).astype(np.float32)
