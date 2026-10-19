"""Review a live session (tools env): what the real-time tool did, what went wrong, where to listen.

  envs/tools/Scripts/python scripts/live_review.py logs/live/<session> [--deep | --no-deep]

Always:
  the session's health (underruns, bursts drained, overloads, errors, engine failures), every line replaced
  with its lateness and overrun, close calls the engine rejected, and every mark you set (the mark hotkey)
  with the lines around it and an input / output clip to listen to.
  Voices recorded with Space: the last stretch of voice that ended before each press, found in a vocal stem
  (the deep check's; without one, the 15 s before every press are separated here in one go with the same
  model - live.ps1 reviews with --no-deep), cut out as a clip, with the line it was if one is known. Presses
  with no voice in the 15 s before (Space is also the game's jump key) are dropped, and presses that land on
  the same voice count once. Without audio: the last line the engine accepted, else the 8 s before.
Deep check (automatic for sessions up to 30 min, --deep for longer ones; much slower than real time):
  the offline pipeline (replace_in_recording.py: vocal stem, verification, claims) runs over the captured
  input and its lines are compared with the live decisions: lines live missed, lines live replaced that
  offline does not confirm, lines replaced late. Each comes with a clip.
Writes <session>/review/review.html (+ clips/, review.json); open it through the local page server:
  http://localhost:8765/logs/live/<session>/review/review.html
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import live_core as LC
from vl_common import EXP, ROOT, load_lines, read_csv
from vl_paths import EXP_BASE, SET_NAME, char_paths

SR = LC.SR


def clip(src, a, b, dst):
    """[a, b) seconds of a session recording into a wav (zero-padded at the edges)."""
    with sf.SoundFile(src) as f:
        n0, n1 = int(a * SR), int(b * SR)
        f.seek(max(0, n0))
        y = f.read(max(0, n1 - max(0, n0)), dtype="float32", always_2d=True)
    y = np.concatenate([np.zeros((max(0, -n0), 2), np.float32), y])
    sf.write(dst, y, SR, subtype="PCM_16")
    return dst.name


def last_voice(vocals, t, back=15.0, after=0.3, gap=0.8, min_run=0.15):
    """The last stretch of voice in the vocal stem that starts before t: (start, end) in seconds, or None.
    Voice: 10 ms frames above max(-50 dBFS, the window's loudest - 35 dB), gaps under `gap` joined, runs under
    `min_run` dropped. Reads only the window, so long sessions stay cheap."""
    with sf.SoundFile(vocals) as f:
        sr = f.samplerate
        a = max(0.0, t - back)
        f.seek(int(a * sr))
        x = f.read(int((t + after - a) * sr), dtype="float32", always_2d=True).mean(1)
    h = int(0.01 * sr)
    if len(x) < h:
        return None
    db = 20 * np.log10(np.sqrt((x[: len(x) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9)
    on = db > max(-50.0, db.max() - 35)
    runs, start, last = [], None, -99
    for i, v in enumerate(on):
        if v:
            if start is None or i - last > gap / 0.01:
                if start is not None:
                    runs.append((start, last + 1))
                start = i
            last = i
    if start is not None:
        runs.append((start, last + 1))
    runs = [(s, e) for s, e in runs if (e - s) * 0.01 >= min_run and a + s * 0.01 < t]
    if not runs:
        return None
    s, e = runs[-1]
    return round(a + s * 0.01, 2), round(a + e * 0.01, 2)


class Stem:
    """A session's vocal stem, whole (the deep check's) or in pieces: [(session start, session end, start in
    the file)] - the windows before the Space presses, separated together (press_stem)."""

    def __init__(self, path, pieces=None):
        self.path, self.pieces = path, pieces or [(0.0, float("inf"), 0.0)]

    def last_voice(self, t):
        for s0, s1, f0 in self.pieces:
            if s0 <= t <= s1:
                seg = last_voice(self.path, f0 + t - s0, back=min(15.0, t - s0))
                return None if seg is None else (round(s0 + seg[0] - f0, 2), round(s0 + seg[1] - f0, 2))
        return None


def press_stem(ses, out, times, back=15.0, after=0.5, gap=1.0):
    """Vocal stem for the stretches before the presses: [t - back, t + after] merged, joined with `gap` s of
    silence into one file, separated by replace_in_recording.separate (Mel-Band RoFormer, as the deep check)."""
    import os

    from replace_in_recording import separate

    ff = str(ROOT / "tools" / "ffmpeg" / "bin")  # the separator calls ffmpeg; env.ps1 puts it on PATH, bash does not
    if ff.lower() not in os.environ.get("PATH", "").lower():
        os.environ["PATH"] = ff + os.pathsep + os.environ.get("PATH", "")
    iv = []
    for t in sorted(times):
        a, b = max(0.0, t - back), t + after
        if iv and a <= iv[-1][1]:
            iv[-1][1] = max(iv[-1][1], b)
        else:
            iv.append([a, b])
    work = out / "voice_stem"
    work.mkdir(exist_ok=True)
    parts, pieces, pos = [], [], 0.0
    with sf.SoundFile(ses / "input.flac") as f:
        sr = f.samplerate
        for a, b in iv:
            f.seek(int(a * sr))
            y = f.read(int((b - a) * sr), dtype="float32", always_2d=True)
            pieces.append((a, a + len(y) / sr, pos))
            parts += [y, np.zeros((int(gap * sr), y.shape[1]), np.float32)]
            pos += len(y) / sr + gap
    src = work / "windows.wav"
    sf.write(src, np.concatenate(parts), sr, subtype="PCM_16")
    print(f"voices: separating {sum(b - a for a, b in iv):.0f} s before {len(times)} Space presses ...", flush=True)
    separate(src, work)
    return Stem(work / "vocals.wav", pieces)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--deep", action="store_true", help="run the offline check even on a long session")
    ap.add_argument("--no-deep", action="store_true", help="skip the offline check")
    a = ap.parse_args()

    ses = Path(a.session).resolve()
    ev = [json.loads(l) for l in open(ses / "events.jsonl", encoding="utf-8")]
    info = json.loads((ses / "session.json").read_text(encoding="utf-8")) if (ses / "session.json").exists() else {}
    start = next((e for e in ev if e["type"] == "start"), {})
    delay, hop = start.get("delay_ms", 200) / 1000, LC.HOP / SR
    lag = delay - hop                                  # output.flac runs this much behind input.flac
    has_audio = (ses / "input.flac").exists() and (ses / "output.flac").exists()
    out = ses / "review"
    (out / "clips").mkdir(parents=True, exist_ok=True)
    lines = {ln["id"]: ln for ln in load_lines()}
    for spec in start.get("also", []) or []:  # more characters in the session: their ids carry the character
        char, set_name, _, _ = spec.split(":")
        lines.update({f"{char}:{ln['id']}": ln for ln in load_lines(char_paths(char, set_name)["lines"])})

    def pair(t0, t1, tag):
        if not has_audio:
            return {}
        return {"in": "clips/" + clip(ses / "input.flac", t0, t1, out / "clips" / f"{tag}_in.wav"),
                "out": "clips/" + clip(ses / "output.flac", t0 + lag, t1 + lag, out / "clips" / f"{tag}_out.wav"),
                "from": round(t0, 2)}

    acc = [e for e in ev if e["type"] == "accept"]
    health = {k: [e for e in ev if e["type"] == k] for k in
              ("underrun", "drain", "overload", "error", "exception", "resync", "drift", "render_guard")}
    marks = []
    for i, m in enumerate(e for e in ev if e["type"] == "mark"):
        t = m["t"]
        near = [{"t": round(e["t"], 2), "type": e["type"], "id": e.get("id", ""), "late_ms": e.get("late_ms", ""),
                 "text": lines.get(e.get("id", ""), {}).get("ref_text", "")[:30]}
                for e in ev if e["type"] in ("accept", "cut", "stop", "reject") and t - 8 <= e.get("t", -99) <= t + 2]
        marks.append({"t": t, "near": near, **pair(t - 6, t + 3, f"mark{i + 1}")})

    # deep check: the offline pipeline over the captured input
    deep = None
    secs = info.get("seconds", 0)
    if has_audio and not a.no_deep and (a.deep or secs <= 1800):
        name = f"live_{ses.name}"
        work = EXP_BASE / "recordings" / name
        done = work / "events.csv"
        if done.exists() and done.stat().st_mtime > (ses / "input.flac").stat().st_mtime:
            print("deep check: reusing the offline result from an earlier review")
            r = subprocess.CompletedProcess([], 0, "", "")
        else:
            print(f"deep check: offline pipeline over {secs:.0f} s of captured audio (slower than real time) ...",
                  flush=True)
            r = subprocess.run([sys.executable, str(Path(__file__).parent / "replace_in_recording.py"),
                                str(ses / "input.flac"), "--name", name, "--no-uncovered"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0 or not done.exists():
            deep = {"error": (r.stderr or r.stdout)[-2000:]}
        else:
            off = read_csv(work / "events.csv")
            from rt_feasibility import onset
            from vl_common import read_mono

            def onset_s(lid):
                return onset(read_mono(lines[lid]["wav"], 16000)[0]) / 16000

            head16 = {}

            def head(lid):
                """The first 300 ms of a line from its onset (16 kHz), unit norm."""
                if lid not in head16:
                    x = read_mono(lines[lid]["wav"], 16000)[0]
                    o = onset(x)
                    s = x[o:o + 4800]
                    s = np.pad(s, (0, 4800 - len(s)))
                    head16[lid] = s / (np.linalg.norm(s) + 1e-9)
                return head16[lid]

            def same(a_id, b_id):
                """The same line, or the same recording under two ids (the game has some twice, transcribed
                differently: 2026-10-14 L003 / L058): matched starts within +-20 ms, as Bank.SIM."""
                if a_id == b_id:
                    return True
                if lines[a_id]["ref_text"] == lines[b_id]["ref_text"] and lines[a_id]["ref_text"] != "":
                    return True
                return float(np.abs(np.correlate(np.pad(head(a_id), 320), head(b_id), "valid")).max()) >= 0.9

            ref = [{"id": e["id"], "t": float(e["time"]) + onset_s(e["id"]), "replace": e["version"] != "保留原声",
                    "text": e["text"]} for e in off]
            used, rows = set(), []
            for e in acc:
                if ":" in e["id"]:  # another character's line (--also): the offline pipeline covers only this one
                    continue
                m = [i for i, r_ in enumerate(ref) if i not in used and abs(r_["t"] - e["t"]) < 0.06
                     and same(r_["id"], e["id"])]
                if m:
                    used.add(m[0])
                    if e.get("late_ms", 0) > 0 and e.get("replace"):
                        rows.append({"kind": "late", "t": round(e["t"], 2), "id": e["id"],
                                     "detail": f"晚 {e['late_ms']} ms", **pair(e["t"] - 1.5, e["t"] + 2.5,
                                                                                  f"late_{e['id']}_{e['t']:.0f}")})
                elif e.get("replace"):
                    rows.append({"kind": "unconfirmed", "t": round(e["t"], 2), "id": e["id"],
                                 "detail": f"匹配度 {e.get('ncc')}，能量 {e.get('ex')} dB",
                                 **pair(e["t"] - 1.5, e["t"] + 2.5, f"unconf_{e['id']}_{e['t']:.0f}")})
            for i, r_ in enumerate(ref):
                if i not in used and r_["replace"]:
                    rows.append({"kind": "missed", "t": round(r_["t"], 2), "id": r_["id"], "detail": "",
                                 **pair(r_["t"] - 1.5, r_["t"] + 2.5, f"missed_{r_['id']}_{r_['t']:.0f}")})
            rows.sort(key=lambda x: x["t"])
            n_rep = sum(1 for r_ in ref if r_["replace"])
            deep = {"offline_lines": len(ref), "offline_replaced": n_rep, "rows": rows,
                    "missed": sum(r_["kind"] == "missed" for r_ in rows),
                    "unconfirmed": sum(r_["kind"] == "unconfirmed" for r_ in rows),
                    "late": sum(r_["kind"] == "late" for r_ in rows),
                    "page": f"../../../../eval/{SET_NAME}/recording_{name}.html"}

    # voices recorded with Space: the last stretch of voice before each press
    rec = EXP_BASE / "recordings" / f"live_{ses.name}"
    presses = [e for e in ev if e["type"] == "voice_mark"]
    stem = None
    if presses and has_audio:
        whole = rec / "stems" / "vocals.wav"
        if whole.exists() and whole.stat().st_mtime > (ses / "input.flac").stat().st_mtime:
            stem = Stem(whole)
        else:
            try:
                stem = press_stem(ses, out, [e["t"] for e in presses])
            except Exception as ex:  # the review must still come out; fall back to the engine's lines
                print(f"voices: separation failed ({ex}); using the lines the engine accepted")
    def dur(lid):
        try:
            return sf.info(lines[lid]["wav"]).duration
        except (KeyError, RuntimeError):
            return 2.0

    known = []  # (start s, end s, id, text): the lines the offline check found, then the ones the engine accepted
    if deep and "rows" in deep:
        known += [(float(e["time"]), float(e["time"]) + float(e["played_s"] or 0), e["id"], e["text"])
                  for e in read_csv(rec / "events.csv")]
    known += [(e["t"], e["t"] + dur(e["id"]), e["id"], lines.get(e["id"], {}).get("ref_text", "")) for e in acc]
    voices, dropped = [], 0
    for m in presses:
        t = m["t"]
        seg = stem.last_voice(t) if stem else None
        how = "人声分轨"
        if stem and seg is None:  # no voice before it: a jump, not a record
            dropped += 1
            continue
        if seg is None:
            prev = [k for k in known if t - 12 <= k[0] <= t]
            seg, how = ((round(max(prev)[0], 2), t), "实时判断") if prev else ((round(t - 8, 2), t), "按键前 8 秒")
        # a known line overlapping the voice: the whole line (a pause inside it split the stretch: L003 at 79.8 s
        # of session 2026-10-14_00-24-53 came out as its second half)
        hit = sorted((k for k in known if k[0] - 0.3 <= seg[1] and k[1] + 0.3 >= seg[0] and k[0] <= t),
                     key=lambda k: -k[0])
        if hit:
            seg = (round(min(seg[0], hit[0][0]), 2), round(min(t, max(seg[1], hit[0][1])), 2))
        if voices and voices[-1]["start"] == seg[0]:  # the same voice pressed again
            voices[-1]["presses"] += 1
            continue
        voices.append({"t": t, "start": seg[0], "end": seg[1], "how": how, "presses": 1,
                       "id": hit[0][2] if hit else "", "text": hit[0][3] if hit else "",
                       **pair(seg[0] - 0.3, seg[1] + 0.3, f"voice{len(voices) + 1}")})

    data = {"session": ses.name, "info": info, "start": start, "voices": voices, "voices_dropped": dropped,
            "accepted": [{**{k: e.get(k) for k in ("t", "id", "decided_ms", "late_ms", "ncc", "ex", "replace")},
                          "text": lines.get(e["id"], {}).get("ref_text", "")[:40]} for e in acc],
            "cuts": [e for e in ev if e["type"] in ("cut", "stop") and e.get("overrun_ms", 0) > 0],
            "rejects": [e for e in ev if e["type"] == "reject"],
            "health": {k: v for k, v in health.items()}, "marks": marks, "deep": deep,
            "audio": {"in": "../input.flac", "out": "../output.flac", "lag": lag} if has_audio else None}
    (out / "review.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tpl = (Path(__file__).parent / "templates" / "live_review.html").read_text(encoding="utf-8")
    (out / "review.html").write_text(tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False)), encoding="utf-8")
    rel = (out / "review.html").relative_to(ROOT).as_posix()
    print(f"replaced {sum(1 for e in acc if e.get('replace'))} lines; marks {len(marks)}; "
          f"voices recorded {len(voices)} ({dropped} presses with no voice before them dropped); "
          f"underruns {len(health['underrun'])}, drains {len(health['drain'])}, overloads {len(health['overload'])}, "
          f"errors {len(health['error']) + len(health['exception'])}")
    if deep and "rows" in deep:
        print(f"deep check: offline found {deep['offline_replaced']} lines to replace; live missed {deep['missed']}, "
              f"unconfirmed {deep['unconfirmed']}, late {deep['late']}")
    elif deep:
        print("deep check failed:", deep.get("error", "")[-500:])
    print(f"-> http://localhost:8765/{rel}")


if __name__ == "__main__":
    main()
