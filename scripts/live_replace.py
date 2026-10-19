"""Live replacement (tools env): the game's sound in -> the character's lines replaced -> your headphones.

The game's output is switched automatically (app_route.py): to a device you do not listen to, e.g.
"扬声器 (Steam Streaming Speakers)" or "CABLE Input" (VB-CABLE), as soon as the game runs, and back to where it was
when the tool stops (if the game was closed first: at its next start). The tool captures that device and plays the
result on your headphones. --route off: leave the game's output alone and set it yourself in Settings > System >
Sound > Volume mixer (应用音量和设备首选项). Nothing in the game is touched either way.
Run:
  envs/tools/Scripts/python scripts/live_replace.py --capture "Steam Streaming Speakers" --play "Headphones"
  envs/tools/Scripts/python scripts/live_replace.py --list                       (devices)
  --also <char>:all:<voice>:final_gsv_a_48k   another character replaced in the same session (repeatable; its line ids
                                           read "other:L001" in the logs; live.ps1 adds the second character by default)
Hotkeys (system-wide, no keyboard hook):
  ~           mark a moment that sounded wrong (logged with its time, for live_review.py); only watched, so
              the game still gets the key
  Space       record the voice that just went by (live_review.py cuts out the last stretch of voice before the
              press); only watched too - it is also the game's jump key, so presses with no voice before them
              are dropped in the review
  Ctrl+Alt+1..9  the n-th character (the current one first, then each --also) in or out of the party: a character
              switched off is not matched at all (no cost); kept in logs/live/party.json for the next session,
              which the launcher page can also write while the tool runs
  Ctrl+Alt+B  bypass on/off (hear the game unchanged, same delay)
  Ctrl+Alt+Q  quit
Each session writes logs/live/<time>/: events.jsonl (every decision, stop, error, mark, overload),
input.flac (the game as captured), output.flac (what you heard) and session.json.
Review it later:  envs/tools/Scripts/python scripts/live_review.py logs/live/<time>

File mode (testing and timing, same engine): --file <recording.wav|.mp4> [--out out.wav]
"""
import os

# One math thread per process (set before numpy loads): the per-bin fits are tiny, and BLAS's pool of 12
# spinning threads took CPU from the game and reserved ~0.8 GB per process (numpy + scipy's own copy).
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import ctypes
import json
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

import app_route as AR
import live_core as LC
from vl_common import EXP, ROOT
from vl_paths import EXP_BASE, VOICE, VOICE_PATH_FIR, voice_cfg  # VL_VOICE picks the replacement voice (live.ps1 -Voice)

SR = LC.SR
BLOCK = 480          # 10 ms device blocks
LOGS = ROOT / "logs" / "live"
PARTY = LOGS / "party.json"  # {"off": [characters left out]}: kept between sessions, read while running


def party_off():
    """The characters switched off (not in the party): their lines are not matched, which saves the matching cost."""
    try:
        return set(json.loads(PARTY.read_text(encoding="utf-8")).get("off", []))
    except (OSError, ValueError):
        return set()


def save_party_off(off):
    PARTY.parent.mkdir(parents=True, exist_ok=True)
    PARTY.write_text(json.dumps({"off": sorted(off)}, ensure_ascii=False), encoding="utf-8")


class Session:
    """Session folder: events.jsonl written as things happen (a crash keeps everything before it)."""

    def __init__(self, tag=""):
        self.dir = LOGS / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + tag)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.f = open(self.dir / "events.jsonl", "a", encoding="utf-8")
        self.lock = threading.Lock()
        self.counts = {}
        self.t0 = time.time()

    def log(self, ev):
        ev = {"wall": round(time.time() - self.t0, 3), **ev}
        with self.lock:
            self.f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            self.f.flush()
            self.counts[ev["type"]] = self.counts.get(ev["type"], 0) + 1
        if ev["type"] in ("exception", "error"):
            print(f"\n[{ev['type']}] {ev.get('error', '')}", file=sys.stderr)

    def close(self, summary):
        self.log({"type": "end"})
        (self.dir / "session.json").write_text(json.dumps({**summary, "counts": self.counts}, ensure_ascii=False,
                                                          indent=1), encoding="utf-8")
        self.f.close()


def find_path_fir():
    p = VOICE_PATH_FIR                                  # this character's, else the first character's (same mixer)
    if p.exists():
        return p
    found = sorted((EXP_BASE / "recordings").glob("*/voice_path_fir.npy"), key=lambda q: q.stat().st_mtime)
    return found[-1] if found else None


def cpu_stats(eng):
    c = np.array(eng.cpu) * 1000 if eng.cpu else np.zeros(1)
    return {"step_ms_p50": round(float(np.median(c)), 2), "step_ms_p95": round(float(np.percentile(c, 95)), 2),
            "step_ms_max": round(float(c.max()), 2), "steps": len(eng.cpu)}


# --- file mode --------------------------------------------------------------------------------------------
def run_file(a, bank):
    src = Path(a.file)
    wav = src
    if src.suffix.lower() not in (".wav", ".flac"):
        wav = Path(LOGS.parent / "tmp_live_input.wav")
        wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "2", "-ar", str(SR),
                        "-c:a", "pcm_f32le", str(wav)], check=True)
    x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    if sr != SR:
        raise SystemExit(f"{wav} is {sr} Hz; give a 48 kHz file (or a video: it is converted)")
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    ses = Session("_file")
    ses.log({"type": "start", "mode": "file", "file": str(src), "delay_ms": a.delay, "rule": list(LC.RULE),
             "also": a.also, "off": a.off})
    eng = LC.Engine(bank, a.delay, log=ses.log)
    if a.off:  # file mode leaves party.json alone (repeatable runs): only --off counts
        eng.set_chars([c for c in bank.chars if c not in set(a.off)])
    t0 = time.perf_counter()
    out = [eng.push(x[i:i + BLOCK]) for i in range(0, len(x) - BLOCK + 1, BLOCK)]
    wall = time.perf_counter() - t0
    y = np.concatenate(out)
    D = eng.D - LC.HOP                                    # output sample j is input sample j - D + HOP
    aligned = np.zeros_like(x)
    aligned[:len(y) - D] = y[D:]                          # undo the delay, to compare with the input
    outp = Path(a.out) if a.out else ses.dir / "output_aligned.wav"
    sf.write(outp, aligned, SR, subtype="PCM_16")
    st = cpu_stats(eng)
    rt = wall / (len(x) / SR)
    ses.close({"mode": "file", "file": str(src), "seconds": len(x) / SR, "real_time_factor": round(rt, 3), **st})
    acc = [json.loads(l) for l in open(ses.dir / "events.jsonl", encoding="utf-8")]
    print(f"{len(x) / SR:.1f} s processed in {wall:.1f} s (real-time factor {rt:.2f}); "
          f"step p50 {st['step_ms_p50']} ms, p95 {st['step_ms_p95']} ms, max {st['step_ms_max']} ms (budget 20 ms)")
    for e in acc:
        if e["type"] in ("accept", "cut", "stop", "exception"):
            print("  " + " ".join(f"{k}={v}" for k, v in e.items() if k not in ("wall", "trace")))
    print(f"-> {outp}\n-> {ses.dir}")


# --- device mode ------------------------------------------------------------------------------------------
class Hotkeys(threading.Thread):
    """Mark (~) and voice (Space, 2026-10-15: 用于记录刚刚过去的语音): single keys, watched (their state polled
    every 15 ms) rather than taken, so the game and other programs still get them. Bypass and quit: Ctrl+Alt+<key>
    via RegisterHotKey (a message loop in this thread), per action the first combination not already taken by
    another program."""
    WANT = {"bypass": "BV", "quit": "QX"}
    WATCH = {0xC0: ("mark", "~"), 0x20: ("voice", "Space")}  # VK_OEM_3 (the `~ key), VK_SPACE

    def __init__(self, on_key, n_chars=0):
        super().__init__(daemon=True)
        self.on_key = on_key
        self.n_chars = n_chars                              # Ctrl+Alt+1..n: a character in or out of the party
        self.got = {}
        self.ready = threading.Event()

    def _watch(self):
        get = ctypes.windll.user32.GetAsyncKeyState
        get.restype = ctypes.c_short
        down = dict.fromkeys(self.WATCH, False)
        while True:
            for vk, (action, _) in self.WATCH.items():
                now = get(vk) < 0                           # high bit: held right now
                if now and not down[vk]:                    # one event per press
                    self.on_key(action)
                down[vk] = now
            time.sleep(0.015)

    def run(self):
        user32 = ctypes.windll.user32
        threading.Thread(target=self._watch, daemon=True).start()
        self.got.update({action: name for action, name in self.WATCH.values()})
        ids = {}
        for action, keys in self.WANT.items():
            for ch in keys:
                hid = len(ids) + 1
                if user32.RegisterHotKey(None, hid, 0x0001 | 0x0002 | 0x4000, ord(ch)):  # Alt | Ctrl | NoRepeat
                    ids[hid] = action
                    self.got[action] = f"Ctrl+Alt+{ch}"
                    break
        for i in range(1, min(self.n_chars, 9) + 1):
            hid = len(ids) + 1
            if user32.RegisterHotKey(None, hid, 0x0001 | 0x0002 | 0x4000, ord(str(i))):
                ids[hid] = f"char{i}"
                self.got[f"char{i}"] = f"Ctrl+Alt+{i}"
        self.ready.set()
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            if msg.message == 0x0312 and msg.wParam in ids:      # WM_HOTKEY
                self.on_key(ids[msg.wParam])


class Fifo:
    """Output samples waiting for the player, with drift control between the two device clocks."""

    def __init__(self):
        self.q = deque()
        self.n = 0
        self.lock = threading.Lock()

    def put(self, x):
        with self.lock:
            self.q.append(x)
            self.n += len(x)

    def take(self, n):
        out, got = [], 0
        with self.lock:
            while got < n and self.q:
                x = self.q.popleft()
                need = n - got
                if len(x) > need:
                    self.q.appendleft(x[need:])
                    x = x[:need]
                out.append(x)
                got += len(x)
            self.n -= got
        y = np.concatenate(out) if out else np.zeros((0, 2), np.float32)
        return y, got



class Drift:
    """Absorbs the clock difference between the capture and playback devices by resampling the output a
    hair (Catmull-Rom; ratio within +-0.5 %, under 9 cents: inaudible), steering the playback backlog to its
    target. Capture arrives in bursts (+-40 ms swings), so the control uses the 2 s median, updated each 1 s."""

    KP, KI, LIMIT = 4e-7, 2e-8, 0.005                       # per sample of backlog error; +-0.5 % (8.7 cents)

    def __init__(self, target, log):
        self.target, self.log = target, log
        self.step = 1.0                                     # input samples per output sample
        self.q = 1.0
        self.tail = np.zeros((3, 2), np.float32)
        self.hist, self.last, self.integral, self.logged = [], time.time(), 0.0, 1.0

    def reset(self):
        self.step, self.integral, self.hist, self.logged = 1.0, 0.0, [], 1.0

    def process(self, y):
        buf = np.concatenate([self.tail, y])
        top = len(buf) - 2
        n = int(np.floor((top - self.q) / self.step)) + 1 if top > self.q else 0
        q = self.q + self.step * np.arange(n)
        i = q.astype(int)
        f = (q - i)[:, None]
        p0, p1, p2, p3 = buf[i - 1], buf[i], buf[i + 1], buf[np.minimum(i + 2, len(buf) - 1)]
        out = 0.5 * (2 * p1 + (p2 - p0) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f ** 2 +
                     (3 * p1 - p0 - 3 * p2 + p3) * f ** 3)
        self.q = (q[-1] + self.step if n else self.q) - (len(buf) - 3)
        self.tail = buf[-3:]
        return out.astype(np.float32)

    def update(self, backlog, t):
        """PI control on the 3 s median backlog, once a second: a steady clock offset ends up in the integral
        (a constant ratio), the bursts of capture average out."""
        self.hist.append(backlog)
        now = time.time()
        if now - self.last < 1.0:
            return
        dt, self.last = now - self.last, now
        self.hist = self.hist[-300:]
        err = float(np.median(self.hist)) - self.target
        self.excess = err if err > 0.1 * SR else 0.0          # a burst: run_live drops it in one go
        self.integral = float(np.clip(self.integral + err * dt, -self.LIMIT / self.KI, self.LIMIT / self.KI))
        self.step = 1.0 + float(np.clip(self.KP * err + self.KI * self.integral, -self.LIMIT, self.LIMIT))
        if abs(self.step - self.logged) > 5e-4:
            self.logged = self.step
            self.log({"type": "drift", "t": round(t, 2), "backlog_ms": round((err + self.target) / SR * 1000),
                      "ratio_ppm": round((self.step - 1) * 1e6)})


CAP_BUFFER = 0.2     # s of WASAPI capture buffer
IDLE_FILL = 0.1      # s with no packet at all before the recorder stands zeros in for the time waited


def steady_capture(log):
    """soundcard's recorder, made to keep the stream's timeline when this process is held up. 2026-10-18: the
    party panel's screen grab held the GIL for 56-115 ms every 2 s, and the second character's lines were lost after their
    first second (the user: 只有开头成功，之后会变回原音; 有一定的噪点):
    - its WASAPI buffer was one 10 ms block (the recorder's blocksize), so a capture thread held up longer lost
      audio: capture() opens it with CAP_BUFFER (still read BLOCK at a time; the latency is unchanged);
    - while no packet was available it returned zeros for the time waited once that passed 4 device periods
      (~40 ms), also when the wait was this thread being held up and the packets were there: the sound got
      73-79 ms holes (146 in 327 s; 0-9 in 10 min before) and the line being replaced moved ~30 ms later each
      time, which tracking could not follow. Zeros now stand in only when nothing has come for IDLE_FILL s,
      checked again after every sleep (a loopback delivers no packets while nothing plays at all)."""
    import soundcard.mediafoundation as mf
    rec = getattr(mf, "_Recorder", None)
    orig = getattr(rec, "_record_chunk", None)
    if orig is None or getattr(orig, "steady", False) or not hasattr(rec, "_capture_available_frames"):
        if orig is None or not getattr(orig, "steady", False):
            log({"type": "error", "where": "capture", "error": "soundcard internals changed: recorder not patched"})
        return

    def _record_chunk(self):
        while self._capture_available_frames() == 0:
            if self._idle_start_time is None:
                self._idle_start_time = time.perf_counter_ns()
            time.sleep(0.002)
            if self._capture_available_frames():
                break
            waited = time.perf_counter_ns() - self._idle_start_time
            if waited > IDLE_FILL * 1e9:
                self._idle_start_time += waited
                return np.zeros(int(self.samplerate * waited / 1e9) * len(set(self.channelmap)), np.float32)
        return orig(self)                              # a packet is there: soundcard reads it (flags and all)

    _record_chunk.steady = True
    rec._record_chunk = _record_chunk


def raise_priority():
    """Above-normal priority for this process: with the game at full load the engine's 20 ms steps took up to
    300 ms and the playback buffer ran dry. Returns True if Windows accepted it."""
    k32 = ctypes.windll.kernel32
    return bool(k32.SetPriorityClass(k32.GetCurrentProcess(), 0x8000))  # ABOVE_NORMAL_PRIORITY_CLASS


def keep_in_ram(extra=1.05):
    """Hard minimum working set at the current size: Windows keeps this process's pages in RAM. With the
    game holding ~7 GB, the engine's pages were swapped out and a 20 ms step stalled for 300 ms (dropouts).
    Returns (ok, MB)."""
    import psutil

    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessWorkingSetSizeEx.argtypes = [wintypes.HANDLE, ctypes.c_size_t, ctypes.c_size_t, wintypes.DWORD]
    ws = psutil.Process().memory_info().rss
    ok = k32.SetProcessWorkingSetSizeEx(k32.GetCurrentProcess(), int(ws * extra), int(ws * 2), 0x1 | 0x8)
    return bool(ok), round(ws / 2**20)  # QUOTA_LIMITS_HARDWS_MIN_ENABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE


def engine_main(final, path, delay, q_in, q_out, also=()):
    """The engine in its own process: the capture and playback threads of the main process then never wait
    for the processing (one Python interpreter lock each)."""
    raise_priority()
    try:
        bank = LC.Bank(final, Path(path) if path else None, log=lambda e: q_out.put(("log", e)), also=also)
        eng = LC.Engine(bank, delay, log=lambda e: q_out.put(("log", e)))
        locked, mb = keep_in_ram()
        q_out.put(("log", {"type": "memory", "engine_mb": mb, "kept_in_ram": locked}))
        q_out.put(("ready", {"lines": len(bank.ids), "voice_path": path, "chars": bank.chars}))
        while True:
            msg = q_in.get()
            if msg is None:
                break
            kind, val = msg
            if kind == "audio":
                block, fast = val
                n0 = len(eng.cpu)
                y = eng.push(block, fast=fast)
                act = [bank.ids[x["k"]] for x in eng.active if x["end"] is None and not x["keep"]]
                q_out.put(("audio", (y, eng.cpu[n0:], act)))
            elif kind == "bypass":
                eng.bypass = val
            elif kind == "chars":
                eng.set_chars(val)
    except Exception as e:
        import traceback
        q_out.put(("log", {"type": "error", "where": "engine", "error": repr(e),
                           "trace": traceback.format_exc(limit=8)}))
    q_out.put(("exit", None))


def device_main(process, capture, play, period, q, route=False):
    """Device checks in their own process: where the game plays and where you listen, every `period` s, sent
    to the main process as plain data (the enumeration is slow Python + COM; kept away from the audio threads).
    route: also keep the game's output on the virtual device (app_route.py)."""
    import live_devices as LD
    router = None
    if route:
        try:
            router = AR.Router(process)
        except Exception as e:  # noqa: BLE001
            q.put(("error", f"automatic game output switching unavailable: {e!r}"))
    while True:
        try:
            outs = LD.outputs()
            if router:
                target, _ = LD.choose_capture(outs, "-", capture)   # the virtual device the game should play to
                changes = router.ensure(target, outs)
                if changes:
                    q.put(("route", changes))
                    time.sleep(0.5)                     # Windows moves the game's stream: look again after it
                    outs = LD.outputs()
            c, why = LD.choose_capture(outs, process, capture)
            mis = LD.misrouted(c)
            game_dev = c.name if mis else ""
            if mis:                                         # keep capturing a virtual device meanwhile
                c, why = LD.choose_capture(outs, "-", capture)
            pl, whyp = LD.choose_play(outs, c, play)
            vol = LD.endpoint_volume(c.id) if c else (0.0, False)
            q.put(("devices", {"cap": (c.id, c.name, why) if c else None, "mis": mis, "game_dev": game_dev,
                               "cap_db": vol[0], "cap_mute": vol[1],
                               "play": (pl.id, pl.name, whyp) if pl else None, "play_why": whyp,
                               "ids": [o.id for o in outs]}))
        except Exception as e:
            q.put(("error", repr(e)))
        time.sleep(period)


def give_back(a, log=None):
    """The game's output back to where it was (now, or at its next start); the live tool is done."""
    AR.LIVE_PID.unlink(missing_ok=True)
    if a.route != "auto":
        return
    try:
        r, back = AR.restore("live tool stopped", log=log)
        if r == "restored":
            print(f"\n{a.process} output given back: {back}")
        elif r == "pending":
            print(f"\n{a.process} is closed: it gets {back} back at its next start "
                  "(a small background watcher waits for it, then exits)")
        elif r == "failed":
            print(f"\n! {a.process} did not take its output back: set it in Windows' volume mixer")
    except Exception as e:  # noqa: BLE001
        if log:
            log({"type": "error", "where": "route", "error": repr(e)})
        print(f"\n! could not give {a.process} its output back ({e!r}): set it in Windows' volume mixer")


def run_live(a, path):
    import multiprocessing as mp
    import soundcard as sc
    import live_devices as LD

    raise_priority()  # capture and playback threads (the engine process raises its own)
    AR.LIVE_PID.parent.mkdir(parents=True, exist_ok=True)
    AR.LIVE_PID.write_text(str(os.getpid()))         # a pending switch-back waits while this runs

    outs = LD.outputs()
    routed = []
    if a.route == "auto":                            # the game to the virtual device now, if it already runs
        try:
            target, _ = LD.choose_capture(outs, "-", a.capture)
            routed = AR.Router(a.process).ensure(target, outs)
            for ev in routed:
                print(f"{a.process} output -> {ev['to']} (switched automatically; given back when the tool stops)")
            if routed:
                time.sleep(0.5)
                outs = LD.outputs()
        except Exception as e:  # noqa: BLE001
            print(f"! automatic game output switching unavailable ({e!r}): set it in Windows' volume mixer")
            a.route = "off"
    cap_o, why_c = LD.choose_capture(outs, a.process, a.capture)
    paused = False
    if LD.misrouted(cap_o) and a.route == "auto":    # not moved yet: the device process keeps at it
        print(f"{a.process} still plays to {cap_o.name}: moving it to the virtual device ...")
        cap_o, why_c = LD.choose_capture(outs, "-", a.capture)
        paused = True
    elif LD.misrouted(cap_o):                        # the game plays straight to a device you hear
        print(f"! {a.process} plays straight to {cap_o.name}: you would hear it twice. Set its output to "
              "Steam Streaming Speakers (Windows Settings > System > Sound > Volume mixer). Waiting for that.")
        cap_o, why_c = LD.choose_capture(outs, "-", a.capture)
        paused = True
    if cap_o is None:
        raise SystemExit(f"capture: {why_c}\n{LD.describe(outs, None, None)}")
    play_o, why_p = LD.choose_play(outs, cap_o, a.play)
    if play_o is None:
        raise SystemExit(f"play: {why_p}\n{LD.describe(outs, cap_o, None)}")
    print(LD.describe(outs, cap_o, play_o))
    cap_db, cap_mute = LD.endpoint_volume(cap_o.id)
    print(f"capture device volume {cap_db:.1f} dB: added back, so you hear the game as loud as before"
          f"{' (it is MUTED: unmute it, nothing reaches the tool)' if cap_mute else ''}")
    dev = {"cap": cap_o, "play": play_o, "cap_gen": 0, "play_gen": 0, "paused": paused,
           "gain_db": -cap_db, "gain": 1.0}
    ses = Session()
    ses.log({"type": "start", "mode": "live", "capture": cap_o.name, "capture_reason": why_c, "play": play_o.name,
             "play_reason": why_p, "process": a.process, "delay_ms": a.delay, "buffer_ms": a.buffer,
             "capture_volume_db": round(cap_db, 1), "capture_muted": cap_mute,
             "rule": list(LC.RULE), "voice_path": str(path), "voice": VOICE, "final": str(EXP / a.final),
             "also": a.also, "route": a.route, "capture_buffer_ms": round(CAP_BUFFER * 1000),
             "idle_fill_ms": round(IDLE_FILL * 1000)})
    steady_capture(ses.log)
    for ev in routed:
        ses.log(ev)
    print(f"voice: {voice_cfg()['label']} ({EXP / a.final})" + "".join(f"\nalso:  {s}" for s in a.also))
    ctx = mp.get_context("spawn")
    q_in, q_out = ctx.Queue(), ctx.Queue()
    proc = ctx.Process(target=engine_main, args=(a.final, str(path) if path else "", a.delay, q_in, q_out, a.also),
                       daemon=True)
    proc.start()
    dq = ctx.Queue()
    dproc = ctx.Process(target=device_main, args=(a.process, a.capture, a.play, a.device_check, dq,
                                                  a.route == "auto"), daemon=True)
    dproc.start()
    t0 = time.time()
    print("loading the lines in the engine process ...", flush=True)
    while True:
        try:
            kind, val = q_out.get(timeout=120)
        except queue.Empty:
            raise SystemExit("engine did not start within 120 s")
        if kind == "ready":
            print(f"engine ready: {val['lines']} lines, voice path {val['voice_path'] or 'none'} "
                  f"({time.time() - t0:.1f} s)")
            chars = val["chars"]
            break
        if kind == "log":
            ses.log(val)
            if val.get("type") == "error":
                raise SystemExit(f"engine failed to start: {val['error']}")

    fin = sf.SoundFile(ses.dir / "input.flac", "w", SR, 2, "PCM_16") if not a.no_record else None
    fout = sf.SoundFile(ses.dir / "output.flac", "w", SR, 2, "PCM_16") if not a.no_record else None
    in_q, fifo = queue.Queue(), Fifo()
    drift = Drift(int(a.buffer * SR / 1000), ses.log)
    D = int(a.delay * SR / 1000)
    lag, cpu = [], []
    stop = threading.Event()
    state = {"marks": 0, "voices": 0, "underruns": 0, "overload": False, "overloads": 0, "t_in": 0, "t_back": 0, "u_win": 0,
             "in_underrun": False, "level_db": -120.0, "bypass": False, "dead": False, "playing": [], "played": 0}
    rawq = deque()                                   # input kept for pass-through if the engine dies

    def apply_party(off, why):
        """Characters switched off (party.json, Ctrl+Alt+<n>, --off): only the others' lines are matched."""
        state["off"] = set(off) & set(chars)
        q_in.put(("chars", [c for c in chars if c not in state["off"]]))
        ses.log({"type": "party", "t": round(state["t_in"] / SR, 2), "off": sorted(state["off"]), "why": why})
        print("\n[characters: " + ", ".join(f"{c} {'OFF' if c in state['off'] else 'on'} (Ctrl+Alt+{i + 1})"
                                             for i, c in enumerate(chars)) + "]")

    apply_party(party_off() | set(a.off), "start")
    state["party_seen"] = PARTY.stat().st_mtime_ns if PARTY.exists() else 0

    def party_watch():
        """The party read off the screen every 2 s (party_screen.py); a reading counts once it repeats, and acts
        only when it differs from the last one acted on - a Ctrl+Alt+<n> change stays until the party changes."""
        import party_screen as PS

        try:
            rd = PS.Reader()
        except (OSError, ValueError) as e:
            ses.log({"type": "party_screen", "error": f"no templates: {e!r}"})
            return
        known = [c for c in chars if c in rd.chars]
        ses.log({"type": "party_screen", "chars": known, "screen": list(PS.screen_size())})
        fw, black = PS.Follow(known), False
        while not stop.is_set():
            try:
                img, origin = PS.grab()
                if not np.asarray(img).any():
                    if not black:
                        black = True
                        ses.log({"type": "party_screen", "warning": "the captured strip is black (exclusive fullscreen?)"})
                    raise ValueError("black")
                r = rd.read(img, origin)
                act = fw.step(rd.party(r))
                if act is not None:
                    ses.log({"type": "party_screen", "t": round(state["t_in"] / SR, 2), "in": sorted(act[0]),
                             "out": sorted(act[1]), "slots": r["slots"]})
                    off = (state["off"] | act[1]) - act[0]
                    if off != state["off"]:
                        save_party_off(off)
                        state["party_seen"] = PARTY.stat().st_mtime_ns
                        apply_party(off, "screen")
            except ValueError:
                pass
            except Exception as e:  # never let the screen reading take the tool down
                ses.log({"type": "party_screen", "error": repr(e)})
            stop.wait(2.0)

    if a.party_screen == "auto" and (ROOT / "data" / "party_ui" / "templates.json").exists():
        threading.Thread(target=party_watch, daemon=True).start()

    def on_key(k):
        t = state["t_in"] / SR - a.delay / 1000      # what you are hearing now, in recording time
        if k.startswith("char"):                     # Ctrl+Alt+<n>: the n-th character in or out of the party
            c = chars[int(k[4:]) - 1]
            off = state["off"] ^ {c}
            save_party_off(off)
            state["party_seen"] = PARTY.stat().st_mtime_ns
            apply_party(off, "hotkey")
        elif k == "mark":
            state["marks"] += 1
            ses.log({"type": "mark", "t": round(t, 2), "playing": state["playing"]})
            print(f"\n[mark {state['marks']} at {t:.1f} s]")
        elif k == "voice":
            state["voices"] += 1
            ses.log({"type": "voice_mark", "t": round(t, 2), "playing": state["playing"]})
            print(f"\n[voice {state['voices']} at {t:.1f} s]")
        elif k == "bypass":
            state["bypass"] = not state["bypass"]
            q_in.put(("bypass", state["bypass"]))
            ses.log({"type": "bypass", "t": round(t, 2), "on": state["bypass"]})
            print(f"\n[bypass {'ON: original voice' if state['bypass'] else 'OFF: replacing'}]")
        elif k == "quit":
            stop.set()

    def capture():
        """Records the loopback of dev["cap"]; reopens when the watcher moves it, retries if it vanishes."""
        while not stop.is_set():
            gen, o = dev["cap_gen"], dev["cap"]
            try:
                with sc.get_microphone(o.id, include_loopback=True).recorder(samplerate=SR, channels=2,
                                                                             blocksize=int(CAP_BUFFER * SR)) as rec:
                    while not stop.is_set() and dev["cap_gen"] == gen:
                        in_q.put(rec.record(numframes=BLOCK))
            except Exception as e:
                ses.log({"type": "error", "where": "capture", "device": o.name, "error": repr(e)})
                stop.wait(1.0)                              # the device went away: the watcher picks another

    def watch():
        """Applies the device process's reports: capture and playback follow the game and your outputs."""
        warned, reopened = 0.0, time.time()
        while not stop.is_set():
            if a.test_reopen and time.time() - reopened > a.test_reopen:   # tests: exercise the switch path
                reopened = time.time()
                dev["cap_gen"] += 1
                dev["play_gen"] += 1
                ses.log({"type": "device", "role": "test", "to": "reopen both"})
            try:
                kind, r = dq.get(timeout=0.5)
            except queue.Empty:
                continue
            if kind == "error":
                ses.log({"type": "error", "where": "devices", "error": r})
                continue
            if kind == "route":                             # the device process moved the game's output
                for ev in r:
                    ses.log(ev)
                    print(f"\n[{a.process} output -> {ev['to']} (automatic)]")
                continue
            if abs(-r["cap_db"] - dev["gain_db"]) > 0.3:
                ses.log({"type": "capture_volume", "db": round(r["cap_db"], 1), "gain_db": round(-r["cap_db"], 1)})
                print(f"\n[capture device volume now {r['cap_db']:.1f} dB: compensated]")
                dev["gain_db"] = -r["cap_db"]
            if r["cap_mute"] and time.time() - warned > 30:
                warned = time.time()
                ses.log({"type": "error", "where": "devices", "error": "capture device muted"})
                print(f"\n[! {dev['cap'].name} is muted: the game does not reach the tool. Unmute it.]")
            mis = r["mis"]
            if mis and time.time() - warned > 30:
                warned = time.time()
                ses.log({"type": "misrouted", "device": r["game_dev"]})
                print(f"\n[! {a.process} plays straight to {r['game_dev']}: replacement paused (you would hear it "
                      "twice). Set its output to Steam Streaming Speakers in Windows' volume mixer.]")
            if dev["paused"] and not mis:
                ses.log({"type": "routed", "device": r["cap"][1] if r["cap"] else ""})
                print(f"\n[{a.process} is routed again: replacing]")
            dev["paused"] = mis
            if r["cap"] and r["cap"][0] != dev["cap"].id:
                c = LD.Out(r["cap"][0], r["cap"][1])
                ses.log({"type": "device", "role": "capture", "from": dev["cap"].name, "to": c.name, "reason": r["cap"][2]})
                print(f"\n[capture -> {c.name} ({r['cap'][2]})]")
                dev["cap"] = c
                dev["cap_gen"] += 1
            if r["play"] is None:
                if dev["play"].id not in r["ids"] and time.time() - warned > 30:
                    warned = time.time()
                    ses.log({"type": "error", "where": "devices", "error": f"no output device to play on ({r['play_why']})"})
                    print(f"\n[! no output device to play on: {r['play_why']}]")
                continue
            if r["play"][0] != dev["play"].id:
                pl = LD.Out(r["play"][0], r["play"][1])
                ses.log({"type": "device", "role": "play", "from": dev["play"].name, "to": pl.name, "reason": r["play"][2]})
                print(f"\n[play -> {pl.name} ({r['play'][2]})]")
                dev["play"] = pl
                dev["play_gen"] += 1

    def playback():
        target = int(a.buffer * SR / 1000)
        last = np.zeros((BLOCK, 2), np.float32)
        ramp = np.linspace(1, 0, BLOCK, dtype=np.float32)[:, None]
        while not stop.is_set():
            gen, o = dev["play_gen"], dev["play"]
            try:
                p_cm = sc.get_speaker(o.id).player(samplerate=SR, channels=2, blocksize=BLOCK)
            except Exception as e:
                ses.log({"type": "error", "where": "playback", "device": o.name, "error": repr(e)})
                stop.wait(1.0)
                continue
            drift.reset()                                   # another device: another clock
            try:
                with p_cm as p:
                    while fifo.n < target and not stop.is_set() and dev["play_gen"] == gen:
                        time.sleep(0.002)
                    while not stop.is_set() and dev["play_gen"] == gen:
                        y, got = fifo.take(BLOCK)
                        if got < BLOCK:                               # nothing processed in time: conceal it
                            state["underruns"] += 1
                            state["u_win"] += 1
                            if not state["in_underrun"]:                  # one entry per episode
                                ses.log({"type": "underrun", "t": round(state["t_in"] / SR, 2),
                                         "input_level_db": state["level_db"]})
                                y = np.concatenate([y, (last * ramp)[got:]])
                            else:
                                y = np.concatenate([y, np.zeros((BLOCK - got, 2), np.float32)])
                            state["in_underrun"] = True
                        elif state["in_underrun"]:                        # back: fade in, no click
                            y = y * ramp[::-1]
                            state["in_underrun"] = False
                        last = y
                        state["played"] += len(y)
                        p.play(y if not dev["paused"] else np.zeros_like(y))
            except Exception as e:
                ses.log({"type": "error", "where": "playback", "device": o.name, "error": repr(e)})
                stop.wait(1.0)                              # unplugged: the watcher picks another

    def deliver(y):
        """Adds back the capture device's volume (ramped over the block), then a soft limiter: a peak over
        -0.4 dBFS pulls the gain down within 2 ms and it recovers over ~100 ms - no hard clipping clicks."""
        target = 10 ** (dev["gain_db"] / 20)
        g = np.linspace(dev["gain"], target, len(y), dtype=np.float32)[:, None]
        dev["gain"] = target
        y = y * g
        peak = float(np.abs(y).max()) if len(y) else 0.0
        lim0 = dev.get("lim", 1.0)
        want = min(1.0, 0.95 / peak) if peak > 0 else 1.0
        lim1 = want if want < lim0 else min(want, lim0 + (1.0 - lim0) * 0.1)  # attack now, release slowly
        if lim0 < 1.0 or lim1 < 1.0:
            ramp = np.full(len(y), lim1, np.float32)
            n = min(len(y), int(0.002 * SR))
            ramp[:n] = np.linspace(lim0, lim1, n)
            y = y * ramp[:, None]
            if lim1 < lim0:
                state["limited"] = state.get("limited", 0) + 1
        dev["lim"] = lim1
        if len(y) and float(np.abs(y).max()) > 1.0:                # only the first 2 ms of an attack can get here
            state["clipped"] = state.get("clipped", 0) + 1
            y = np.clip(y, -1.0, 1.0)
        if fout is not None:
            fout.write(y)
        fifo.put(drift.process(y))

    hk = Hotkeys(on_key, n_chars=len(chars))
    hk.start()
    hk.ready.wait(2)
    keys = ", ".join(f"{v} {k}" for k, v in hk.got.items())
    ses.log({"type": "hotkeys", "keys": hk.got})
    threads = [threading.Thread(target=f, daemon=True) for f in (capture, playback, watch)]
    for th in threads:
        th.start()
    print(f"capturing {dev['cap'].name} -> playing {dev['play'].name} (devices followed automatically), "
          f"delay {a.delay} + {a.buffer} ms. Hotkeys: {keys}. Session: {ses.dir}")
    last_status, last_health, levels = time.time(), time.time(), []
    try:
        t_start = time.time()
        while not stop.is_set():
            if a.seconds and time.time() - t_start > a.seconds:
                break
            if (LOGS / "STOP").exists():                          # a clean stop from outside (a restart)
                (LOGS / "STOP").unlink(missing_ok=True)
                ses.log({"type": "stop_file"})
                break
            m_party = PARTY.stat().st_mtime_ns if PARTY.exists() else 0
            if m_party != state["party_seen"]:                    # party.json changed (the launcher page)
                state["party_seen"] = m_party
                if party_off() != state["off"]:
                    apply_party(party_off(), "party.json")
            try:
                block = in_q.get(timeout=0.5)
            except queue.Empty:
                block = None
            if block is not None and np.any(block):
                state["last_sound"] = time.time()
            if time.time() - state.get("last_sound", t_start) > 10 and not state.get("hinted"):
                state["hinted"] = True                          # a loopback delivers nothing while nothing plays
                ses.log({"type": "no_sound", "t": round(state["t_in"] / SR, 1), "capture": dev["cap"].name})
                print(f"\n[no sound from {dev['cap'].name} for 10 s: is the game running"
                      + ("? (its output is switched here automatically)]" if a.route == "auto" else
                         f" and its output set to it? Windows Settings > System > Sound > Volume mixer > "
                         f"{a.process}]"))
            if block is not None and np.any(block) and state.get("hinted"):
                state["hinted"] = False
            if block is not None:
                state["t_in"] += len(block)
                if fin is not None:
                    fin.write(block)
                levels.append(float(np.sqrt(np.mean(block ** 2)) + 1e-9))
                state["level_db"] = round(20 * np.log10(levels[-1]), 1)
                rawq.append(block)
                if state["dead"]:                               # engine gone: the game, delayed, unchanged
                    while sum(len(b) for b in rawq) > D:
                        deliver(rawq.popleft())
                else:
                    while sum(len(b) for b in rawq) > D + SR:   # keep ~1 s for a switch-over
                        rawq.popleft()
                    inflight = state["t_in"] - state["t_back"] - D
                    # behind: pass audio through until caught up. Capture arrives in bursts of up to ~0.2 s, which
                    # the engine clears in a few ms: only a backlog that keeps growing past 0.4 s is an overload.
                    fast = inflight > 0.4 * SR or (state["overload"] and inflight > 0.1 * SR)
                    if fast != state["overload"]:
                        state["overload"] = fast
                        state["overloads"] += int(fast)
                        ses.log({"type": "overload" if fast else "recovered", "t": state["t_in"] / SR,
                                 "backlog_ms": round(inflight / SR * 1000)})
                    lag.append(max(0, inflight))
                    q_in.put(("audio", (block, fast)))
            while True:                                         # everything the engine has sent back
                try:
                    kind, val = q_out.get_nowait()
                except queue.Empty:
                    break
                if kind == "audio":
                    y, c, state["playing"] = val
                    cpu.extend(c)
                    state["t_back"] += len(y)
                    if len(y):
                        deliver(y)
                elif kind == "log":
                    ses.log(val)
                elif kind == "exit" and not state["dead"]:
                    state["dead"] = True
                    ses.log({"type": "error", "where": "engine",
                             "error": "engine process ended; passing the game through unchanged"})
                    print("\n[engine stopped: the game's sound now passes through unchanged]")
            if not state["dead"] and not proc.is_alive():
                state["dead"] = True
                ses.log({"type": "error", "where": "engine",
                         "error": f"engine process died (exit {proc.exitcode}); passing the game through unchanged"})
                print("\n[engine died: the game's sound now passes through unchanged]")
            drift.update(fifo.n, state["t_in"] / SR)
            if getattr(drift, "excess", 0) > 0:                  # a capture burst left 100+ ms extra: drop it
                drop = int(drift.excess)
                y, got = fifo.take(fifo.n)
                f = BLOCK
                if got > drop + 2 * f:
                    y = np.concatenate([y[:f] * np.linspace(1, 0, f)[:, None] + y[drop:drop + f] *
                                        np.linspace(0, 1, f)[:, None], y[drop + f:]]).astype(np.float32)
                    ses.log({"type": "drain", "t": round(state["t_in"] / SR, 2),
                             "dropped_ms": round(drop / SR * 1000)})
                fifo.put(y)
                drift.excess, drift.hist = 0, []
            if time.time() - last_health > 5:                   # health record for later diagnosis
                c = np.array(cpu[-250:] or [0]) * 1000
                lw = np.array(lag[-500:] or [0]) / SR * 1000
                ses.log({"type": "health", "t": round(state["t_in"] / SR, 1),
                         "step_ms_p95": round(float(np.percentile(c, 95)), 1), "step_ms_max": round(float(c.max()), 1),
                         "engine_backlog_ms_max": round(float(lw.max())),
                         "playback_buffer_ms": round(fifo.n / SR * 1000),
                         "underrun_blocks": state["u_win"], "ratio_ppm": round((drift.step - 1) * 1e6)})
                state["u_win"], last_health = 0, time.time()
            if time.time() - last_status > 2 and levels:
                c = np.array(cpu[-100:] or [0]) * 1000
                print(f"\r{state['t_in'] / SR:7.1f}s  in {20 * np.log10(np.mean(levels)):6.1f} dB  "
                      f"replaced {ses.counts.get('accept', 0):3d}  playing {','.join(state['playing']) or '-':10s} "
                      f"step {np.median(c):.1f}/{c.max():.0f} ms  buffer {fifo.n / SR * 1000:4.0f} ms  "
                      f"-> {dev['play'].name[:24]}  "
                      f"{'BYPASS' if state['bypass'] else ''}{'ENGINE DOWN' if state['dead'] else ''}"
                      f"{'PAUSED (game misrouted)' if dev['paused'] else ''}   ",
                      end="", flush=True)
                last_status, levels = time.time(), []
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        q_in.put(None)
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
        dproc.terminate()
        dproc.join(timeout=2)                               # it may be routing: let it go before switching back
        give_back(a, ses.log)
        for th in threads:
            th.join(timeout=2)
        for f in (fin, fout):
            if f is not None:
                f.close()
        lg = np.array(lag or [0]) / SR * 1000
        c = np.array(cpu or [0]) * 1000
        ses.close({"mode": "live", "seconds": state["t_in"] / SR, "marks": state["marks"], "voices": state["voices"],
                   "underruns": state["underruns"], "overloads": state["overloads"],
                   "engine_backlog_ms_p50": round(float(np.median(lg)), 1),
                   "engine_backlog_ms_p99": round(float(np.percentile(lg, 99)), 1),
                   "engine_backlog_ms_max": round(float(lg.max()), 1),
                   "drift_ratio_ppm": round((drift.step - 1) * 1e6), "engine_died": state["dead"],
                   "gain_db": round(dev["gain_db"], 1), "clipped_blocks": state.get("clipped", 0),
                   "limited_blocks": state.get("limited", 0),
                   "played_vs_captured_ppm": round((state["played"] / max(state["t_in"], 1) - 1) * 1e6),
                   "step_ms_p50": round(float(np.median(c)), 2), "step_ms_p95": round(float(np.percentile(c, 95)), 2),
                   "step_ms_max": round(float(c.max()), 2), "steps": len(cpu)})
        print(f"\nsession saved: {ses.dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list capture and playback devices")
    ap.add_argument("--capture", default="auto",
                    help="output device the game plays to (its loopback is captured); auto: found from the game's "
                         "audio session, followed while running")
    ap.add_argument("--play", default="auto",
                    help="where you listen; auto: the Windows default if it is a real device, else headphones > "
                         "speakers > display > digital out; followed while running (plug in, unplug, change default)")
    ap.add_argument("--process", default=os.environ.get("VL_GAME_PROCESS", ""),
                    help="the app whose sound is captured (its audio session; env VL_GAME_PROCESS)")
    ap.add_argument("--route", choices=("auto", "off"), default=os.environ.get("VL_ROUTE", "auto"),
                    help="auto: switch the game's output to the virtual device while running and back afterwards; "
                         "off: leave it to you (env VL_ROUTE)")
    ap.add_argument("--device-check", type=float, default=2.0, help="s between device checks")
    ap.add_argument("--test-reopen", type=float, default=0, help=argparse.SUPPRESS)  # tests: reopen devices every N s
    ap.add_argument("--delay", type=int, default=200, help=f"ms the sound is held back (min {LC.MIN_DELAY})")
    ap.add_argument("--buffer", type=int, default=80, help="ms of playback buffer against hiccups (adds to delay)")
    ap.add_argument("--final", default="final_48k", help="exp/ group with the replacement lines")
    ap.add_argument("--also", action="append", default=[], metavar="CHAR:SET:VOICE:GROUP",
                    help="another character replaced in the same session, e.g. <char>:all:<voice>:final_gsv_a_48k")
    ap.add_argument("--off", action="append", default=[], metavar="CHAR",
                    help="a character switched off at the start (not in the party), on top of logs/live/party.json")
    ap.add_argument("--party-screen", choices=("auto", "off"), default="auto",
                    help="auto: read the party off the game's party panel every 2 s (party_screen.py, needs "
                         "data/party_ui/templates.json) and switch characters in and out")
    ap.add_argument("--path", default="", help="voice path FIR (.npy); default exp/<set>/voice_path_fir.npy")
    ap.add_argument("--no-record", action="store_true", help="do not keep input/output audio of the session")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this many seconds (tests)")
    ap.add_argument("--file", default="", help="file mode: run over a recording instead of devices")
    ap.add_argument("--out", default="", help="file mode: output wav (default: in the session folder)")
    a = ap.parse_args()

    if a.list:
        import live_devices as LD
        outs = LD.outputs()
        cap, why_c = LD.choose_capture(outs, a.process, a.capture)
        play, why_p = LD.choose_play(outs, cap, a.play)
        print("output devices now connected:")
        print(LD.describe(outs, cap, play))
        print(f"capture: {cap.name if cap else '-'} ({why_c})")
        print(f"play:    {play.name if play else '-'} ({why_p})")
        if LD.misrouted(cap):
            print(f"! {a.process} plays straight to {cap.name}: set its output to Steam Streaming Speakers")
        raise SystemExit
    path = Path(a.path) if a.path else find_path_fir()
    if a.file:
        t0 = time.time()
        bank = LC.Bank(a.final, path, also=a.also, log=lambda e: print(
            f"bank: {e['lines']} lines, voice path {e['voice_path'] or 'none (matching less reliable)'}; "
            f"{e.get('sources')}" if e["type"] == "bank" else f"bank: {e}"))
        print(f"loaded in {time.time() - t0:.1f} s")
        run_file(a, bank)
    else:
        if a.party_screen == "auto":                 # screen coordinates in physical pixels (display scaling)
            import party_screen
            party_screen.dpi_aware()
        try:
            run_live(a, path)
        finally:                                     # also when it stops before the session is running
            st = AR.load()
            if st.get("active") and not st.get("pending"):
                give_back(a)
            AR.LIVE_PID.unlink(missing_ok=True)
