"""Round 4: split the longer source lines into phrases at their natural pauses (tools env).

IndexTTS turns the whole emotion prompt into one vector and keeps the melody alive for 2-4 s at
most, so long lines came out flat. Here each line longer than --min-split seconds is cut where the
text has a break (、。！？ or a space) AND the original take actually pauses; every phrase then gets
its own emotion prompt, target reference and duration target (run_indextts.py --phrases).

Text and audio are tied together with Whisper word timestamps; a text break with no pause nearby
is not used, and phrases shorter than --min-phrase seconds are merged into a neighbour.

Only lines routed to the re-dub (scan_lines.py: speech / refine) are used; non-verbal lines go to
RVC on the original instead. Without a routes.csv every line is used.

Writes <set>/phrases/<line>_<n>.wav and phrases.csv:
  id, line_id, idx, text, cut_start/cut_end (where the phrase wav was cut from the line),
  speech_start/speech_end (its voiced part, seconds into the line)
Usage:  envs/tools/Scripts/python scripts/segment_lines.py
"""
import argparse
import difflib
import unicodedata

import numpy as np
import soundfile as sf

from vl_common import PHRASES, ensure_cuda_dlls, load_lines, normalize_ja, read_mono, write_csv
from vl_paths import routes

OUT = PHRASES
BREAKS = set("、。！？!?…‥ 　")


def silences(wav, sr, min_gap=0.12, floor_db=-35.0):
    """(start, end) of pauses inside the speech: 10 ms frames quieter than peak + floor_db."""
    hop = int(0.01 * sr)
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(wav, (0, hop)), int(0.025 * sr))[::hop]
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    quiet = db < db.max() + floor_db
    voiced = np.flatnonzero(~quiet)
    if len(voiced) == 0:
        return []
    gaps, start = [], None
    for k in range(voiced[0], voiced[-1] + 1):
        if quiet[k] and start is None:
            start = k
        elif not quiet[k] and start is not None:
            if (k - start) * 0.01 >= min_gap:
                gaps.append((start * 0.01, k * 0.01))
            start = None
    out = []  # a breath blip (< 0.1 s of sound) inside one pause must not split it in two
    for g in gaps:
        if out and g[0] - out[-1][1] <= 0.1:
            out[-1] = (out[-1][0], g[1])
        else:
            out.append(g)
    return out


def even_rate_time(ref_n, i, gaps, s0, s1):
    """Where break i falls if the line is spoken at an even mora rate (pauses excluded).
    Fallback when Whisper skipped the words around the break (it often drops a line's first clause)."""
    k_before, k_all = len(normalize_ja(ref_n[:i + 1])), max(1, len(normalize_ja(ref_n)))
    inner = sorted((max(a, s0), min(b, s1)) for a, b in gaps if b > s0 and a < s1)
    target = ((s1 - s0) - sum(b - a for a, b in inner)) * k_before / k_all
    t, acc = s0, 0.0
    for a, b in inner:
        if acc + (a - t) >= target:
            break
        acc, t = acc + (a - t), b
    return t + (target - acc)


def char_times(segments):
    """Spread each Whisper word's time span over its characters -> text, start[], end[], indices of
    every word's last character, and the segment joins as (index of the segment's last char, time).
    Whisper starts a new segment at sentence ends; the transcript joined them without punctuation."""
    text, st, en, word_end, seg_breaks = "", [], [], set(), []
    for s in segments:
        for w in s.words or []:
            chars = unicodedata.normalize("NFKC", w.word).strip()
            if not chars:
                continue
            d = (w.end - w.start) / len(chars)
            for i, c in enumerate(chars):
                text += c
                st.append(w.start + i * d)
                en.append(w.start + (i + 1) * d)
            word_end.add(len(text) - 1)
        if text:
            seg_breaks.append((len(text) - 1, s.end))
    return text, np.array(st), np.array(en), word_end, seg_breaks[:-1]


def sentence_breaks(ref_n, m, seg_breaks, punct):
    """Whisper segment joins as extra breaks (ref index, time), unless punctuation is already within 2 chars."""
    inv = {h: r for r, h in m.items()}
    out = []
    for h, t in seg_breaks:
        r = next((inv[h - d] for d in range(3) if h - d in inv), None)
        if r is not None and 0 < r < len(ref_n) - 1 and all(abs(r - i) > 2 for i, _ in punct):
            out.append((r, t))
    return out


def align(ref_n, hyp):
    """ref char index -> Whisper char index, for the characters both texts share."""
    m = {}
    for a, b, size in difflib.SequenceMatcher(None, ref_n, hyp, autojunk=False).get_matching_blocks():
        for k in range(size):
            m[a + k] = b + k
    return m


def break_times(ref_n, m, st, en):
    """Time of each text break (、。！？ space) in ref_n; None when the words around it were not recognised."""
    out = []
    for i, c in enumerate(ref_n):
        if c not in BREAKS or i == 0 or i == len(ref_n) - 1 or ref_n[i + 1] in BREAKS:
            continue
        before = [j for j in range(i - 1, max(-1, i - 4), -1) if j in m][:1]
        after = [j for j in range(i + 1, min(len(ref_n), i + 4)) if j in m][:1]
        out.append((i, (en[m[before[0]]] + st[m[after[0]]]) / 2 if before and after else None))
    return out


def phrase_cuts(ref_n, breaks, gaps, dur, s0, s1, max_off=0.6):
    """Keep the breaks that sit on a real pause; pieces are [text_from, text_to, cut_start, cut_end]."""
    cuts, used = [], set()
    for i, t in breaks:
        t = t if t is not None else even_rate_time(ref_n, i, gaps, s0, s1)
        near = [g for g in gaps if g not in used and g[0] - max_off <= t <= g[1] + max_off
                and (not cuts or (g[0] + g[1]) / 2 > cuts[-1][1])]
        if near:
            g = min(near, key=lambda g: abs((g[0] + g[1]) / 2 - t))
            used.add(g)
            cuts.append((i, (g[0] + g[1]) / 2))
    pieces, prev_i, prev_t = [], 0, 0.0
    for i, t in cuts:
        pieces.append([prev_i, i + 1, prev_t, t])
        prev_i, prev_t = i + 1, t
    pieces.append([prev_i, len(ref_n), prev_t, dur])
    return pieces


def merge_short(pieces, ref_n, wav, sr, min_phrase):
    """Glue each too-short phrase to the neighbour it is spoken closer to (shorter pause):
    「…」 -> 「…」"""
    while len(pieces) > 1:
        b = [speech_bounds(wav[int(p[2] * sr):int(p[3] * sr)], sr, p[2]) for p in pieces]
        short = [k for k, p in enumerate(pieces)
                 if b[k][1] - b[k][0] < min_phrase or len(ref_n[p[0]:p[1]].strip(" 　")) < 3]
        if not short:
            break
        k = short[0]
        before = b[k][0] - b[k - 1][1] if k > 0 else float("inf")
        after = b[k + 1][0] - b[k][1] if k < len(pieces) - 1 else float("inf")
        j = k - 1 if before <= after else k
        pieces[j:j + 2] = [[pieces[j][0], pieces[j + 1][1], pieces[j][2], pieces[j + 1][3]]]
    return pieces


def split_long(pieces, ref_n, m, en, word_end, gaps, wav, sr, max_phrase, min_phrase):
    """Whisper leaves long stretches of a monologue unpunctuated (a 27 s phrase in a 67 s line), and
    one long phrase goes flat again. Cut those at their longest breathing pauses, at the word end
    Whisper places right before the pause; a pause with no word end within 0.5 s is not used."""
    out, todo = [], list(pieces)
    while todo:
        ia, ib, c0, c1 = p = todo.pop(0)
        s, e = speech_bounds(wav[int(c0 * sr):int(c1 * sr)], sr, c0)
        ends = [i for i in range(ia + 2, ib - 3) if i in m and m[i] in word_end]  # >= 3 chars per side
        if e - s <= max_phrase or not ends:
            out.append(p)
            continue
        inner = [g for g in gaps if s + min_phrase < g[0] and g[1] < e - min_phrase and g[1] - g[0] >= 0.2]
        for g in sorted(inner, key=lambda g: g[0] - g[1]):  # longest pause first
            i = min(ends, key=lambda i: abs(en[m[i]] - g[0]))
            if abs(en[m[i]] - g[0]) <= 0.5:
                mid = (g[0] + g[1]) / 2
                todo[0:0] = [[ia, i + 1, c0, mid], [i + 1, ib, mid, c1]]
                break
        else:
            out.append(p)
    return out


def speech_bounds(seg, sr, offset):
    import librosa

    if len(seg) == 0:
        return offset, offset
    _, (a, b) = librosa.effects.trim(seg, top_db=40)
    return offset + a / sr, offset + b / sr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-split", type=float, default=3.0, help="only lines longer than this are split (s)")
    ap.add_argument("--min-phrase", type=float, default=1.0, help="shortest voiced phrase (s)")
    ap.add_argument("--max-phrase", type=float, default=5.0, help="longer phrases are cut again at pauses (s)")
    a = ap.parse_args()

    ensure_cuda_dlls()
    from faster_whisper import WhisperModel
    from vl_common import CACHE

    asr = WhisperModel("large-v3", device="cuda", compute_type="float16", download_root=str(CACHE / "whisper"))
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.wav"):
        old.unlink()
    rows, route = [], routes()
    for ln in load_lines():
        if route and route.get(ln["id"]) == "nonverbal":
            continue
        wav, sr = read_mono(ln["wav"])
        dur = len(wav) / sr
        ref_n = unicodedata.normalize("NFKC", ln["ref_text"])
        pieces = [[0, len(ref_n), 0.0, dur]]
        if dur > a.min_split:
            segs, _ = asr.transcribe(ln["wav"], language="ja", beam_size=5, word_timestamps=True,
                                     condition_on_previous_text=False, initial_prompt=ln["ref_text"])
            hyp, st, en, word_end, seg_breaks = char_times(list(segs))
            m = align(ref_n, hyp)
            gaps = silences(wav, sr)
            s0, s1 = speech_bounds(wav, sr, 0.0)
            breaks = break_times(ref_n, m, st, en)
            breaks = sorted(breaks + sentence_breaks(ref_n, m, seg_breaks, breaks))
            pieces = merge_short(phrase_cuts(ref_n, breaks, gaps, dur, s0, s1), ref_n, wav, sr, a.min_phrase)
            pieces = split_long(pieces, ref_n, m, en, word_end, gaps, wav, sr, a.max_phrase, a.min_phrase)
        pieces = [[ref_n[ia:ib].strip(" 　"), c0, c1] for ia, ib, c0, c1 in pieces]
        for k, (text, c0, c1) in enumerate(pieces, 1):
            seg = wav[int(c0 * sr):int(c1 * sr)]
            s, e = speech_bounds(seg, sr, c0)
            pid = f"{ln['id']}_{k}"
            sf.write(OUT / f"{pid}.wav", seg, sr, subtype="PCM_16")
            rows.append({"id": pid, "line_id": ln["id"], "idx": k, "text": text,
                         "cut_start": round(c0, 3), "cut_end": round(c1, 3),
                         "speech_start": round(s, 3), "speech_end": round(e, 3)})
        print(f"{ln['id']} ({dur:.1f}s) -> " + " | ".join(f"{p[0]} [{p[1]:.2f}-{p[2]:.2f}]" for p in pieces))
    write_csv(OUT / "phrases.csv", rows)
    print(f"{len(rows)} phrases from {len(set(r['line_id'] for r in rows))} lines -> {OUT / 'phrases.csv'}")
