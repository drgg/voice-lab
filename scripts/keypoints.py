"""Prosodic key points of her real clips that a generated re-dub misses (tools env; VL_VOICE).

  python scripts/keypoints.py <gen dir> --name <system>

Moved here from the session scratchpad (change log 2026-10-15_06) when tts_lines.py's keyed pick needed it.
Per test line (tts_testset.py): pitch (parselmouth, 10 ms) and loudness of the real clip and the generated one,
aligned by DTW on MFCCs. Key points of the real clip:
  peak    a pitch peak standing >= 2.5 st above its surroundings (scipy find_peaks prominence), >= 120 ms apart
  jump    pitch moving >= 4 st within 200 ms inside a voiced run (up or down)
  final   the last 300 ms of voicing: rise or fall >= 2 st
  accent  a loudness burst >= 6 dB above the clip's voiced median
A key point is kept by the generated take when it shows the same kind of event within +-80 ms (aligned) at >= 50 %
of the real one's size. Each key point is tagged with the word Whisper puts there (word timestamps on the real clip)
and its place (start / middle / end of the line). Liveliness per line: pitch spread (10-90 %), pitch speed (90th pct
of |st per 50 ms|), peaks per second, loudness spread. Writes <eval>/keypoints_<name>/ (index.html with both
contours per line, missed points marked) and prints the summary.
"""
import argparse
import json
import re
from pathlib import Path

import librosa
import numpy as np

import prosody
from emotion_match import clip_path
from vl_common import EVAL, TARGET, read_csv, read_mono, whisper_model
from vl_paths import voice_cfg

STEP = 0.01


def tracks(path):
    wav, sr = read_mono(path)
    st, xs = prosody.semitones(wav, sr)
    y = librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav
    h = 160
    fr = y[: len(y) // h * h].reshape(-1, h)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
    mf = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=20, n_fft=400, hop_length=h)
    mf = (mf - mf.mean(1, keepdims=True)) / (mf.std(1, keepdims=True) + 1e-6)
    n = mf.shape[1]
    idx = np.clip(np.round((np.arange(n) * STEP - xs[0]) / STEP).astype(int), 0, len(st) - 1)
    stf = st[idx]
    stf[(np.arange(n) * STEP < xs[0] - STEP) | (np.arange(n) * STEP > xs[-1] + STEP)] = np.nan
    return {"st": stf, "db": db[:n] if len(db) >= n else np.pad(db, (0, n - len(db)), constant_values=-90), "mf": mf}


def smooth(st, k=3):
    out = st.copy()
    ok = np.isfinite(st)
    for i in np.where(ok)[0]:
        w = st[max(0, i - k):i + k + 1]
        out[i] = np.nanmean(w)
    return out


def events(t):
    from scipy.signal import find_peaks

    st, db = smooth(t["st"]), t["db"]
    ev = []
    filled = np.where(np.isfinite(st), st, np.nanmin(st) - 10 if np.isfinite(st).any() else -10)
    pk, prop = find_peaks(filled, prominence=2.5, distance=12)
    for p, pr in zip(pk, prop["prominences"]):
        if np.isfinite(st[p]):
            ev.append(("peak", int(p), float(pr)))
    i, n = 0, len(st)
    while i < n - 20:
        w = st[i:i + 20]
        if np.isfinite(w).all() and abs(w[-1] - w[0]) >= 4:
            ev.append(("jump", i + 10, float(w[-1] - w[0])))
            i += 20
        else:
            i += 1
    v = np.where(np.isfinite(st))[0]
    if len(v) > 30:
        tail = st[v[-30:]]
        d = float(np.nanmedian(tail[-8:]) - np.nanmedian(tail[:8]))
        if abs(d) >= 2:
            ev.append(("final", int(v[-15]), d))
    vm = np.nanmedian(db[np.isfinite(t["st"])]) if np.isfinite(t["st"]).any() else np.median(db)
    pk, prop = find_peaks(db, height=vm + 6, distance=15)
    for p in pk:
        ev.append(("accent", int(p), float(db[p] - vm)))
    return ev


def kept(e, g_ev, warp, tol=8):
    kind, pos, size = e
    gp = warp(pos)
    for k2, p2, s2 in g_ev:
        if k2 == kind and abs(p2 - gp) <= tol and np.sign(s2) == np.sign(size) and abs(s2) >= 0.5 * abs(size):
            return True
    return False


def kept_share(perf, take, perf_tracks=None):
    """Share of a performance's key points that a take of the same line shows, the take aligned to it by DTW on
    MFCCs - the keyed pick (tts_lines.py). perf_tracks: tracks(perf) when already computed."""
    r = perf_tracks or tracks(perf)
    r_ev = events(r)
    if not r_ev:
        return 0.0
    g = tracks(take)
    _, wp = librosa.sequence.dtw(X=r["mf"], Y=g["mf"], metric="cosine")
    r2g = {}
    for i, j in wp[::-1]:
        r2g.setdefault(int(i), int(j))
    g_ev = events(g)
    return float(np.mean([kept(e, g_ev, lambda i: r2g.get(int(i), int(i))) for e in r_ev]))


def lively(t):
    st = t["st"]
    v = st[np.isfinite(st)]
    if len(v) < 10:
        return dict(spread=np.nan, speed=np.nan, peaks=np.nan, loud=np.nan)
    sp = np.abs(st[5:] - st[:-5])
    sp = sp[np.isfinite(sp)]
    n_pk = sum(1 for e in events(t) if e[0] == "peak")
    return dict(spread=float(np.percentile(v, 90) - np.percentile(v, 10)), speed=float(np.percentile(sp, 90)) if len(sp) else np.nan,
                peaks=n_pk / (len(v) * STEP), loud=float(np.std(t["db"][np.isfinite(st)])))


def place(pos, n):
    f = pos / max(n, 1)
    return "start" if f < 0.25 else "end" if f > 0.75 else "middle"


def word_at(words, t):
    for w in words:
        if w["start"] - 0.05 <= t <= w["end"] + 0.05:
            return w["word"]
    return ""


def plot(path, r, g, warp_inv, missed, words, title):
    from PIL import Image, ImageDraw, ImageFont

    W, H, pad = 900, 280, 46
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 14)
    except OSError:
        font = ImageFont.load_default()
    n = len(r["st"])
    allv = np.concatenate([r["st"][np.isfinite(r["st"])], g["st"][np.isfinite(g["st"])]])
    lo, hi = (np.min(allv) - 1, np.max(allv) + 1) if len(allv) else (0, 30)
    X = lambda i: pad + (W - 2 * pad) * i / max(n - 1, 1)  # noqa: E731
    Y = lambda v: H - pad - (H - 2 * pad - 20) * (v - lo) / max(hi - lo, 1e-6)  # noqa: E731
    for semi in range(int(np.ceil(lo)), int(hi) + 1, 2):
        d.line([(pad, Y(semi)), (W - pad, Y(semi))], fill=(235, 235, 235))
    gs = np.full(n, np.nan)  # generated contour mapped onto the real clip's time axis
    for i in range(n):
        j = warp_inv(i)
        if 0 <= j < len(g["st"]):
            gs[i] = g["st"][j]
    for arr, col in ((r["st"], (37, 99, 235)), (gs, (234, 88, 12))):
        prev = None
        for i in range(n):
            if np.isfinite(arr[i]):
                pt = (X(i), Y(arr[i]))
                if prev:
                    d.line([prev, pt], fill=col, width=2)
                prev = pt
            else:
                prev = None
    for kind, pos, size in missed:
        x = X(pos)
        d.line([(x, pad - 10), (x, H - pad)], fill=(220, 38, 38), width=1)
        d.text((x + 2, pad - 12), {"peak": "峰", "jump": "跳", "final": "尾", "accent": "重"}[kind], fill=(220, 38, 38), font=font)
    for w in words:
        x = X(int(w["start"] / STEP))
        d.text((x, H - pad + 6), w["word"].strip(), fill=(60, 60, 60), font=font)
    d.text((pad, 4), title, fill=(30, 30, 30), font=font)
    d.text((pad, 20), "蓝：原声  橙：生成（已对齐到原声时间）  红线：原声有、生成没做出来的关键点", fill=(90, 90, 90), font=font)
    img.save(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_dir")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    test = read_csv(TARGET / "tts" / "testset.csv")
    asr = whisper_model()
    out = EVAL.parent / f"{TARGET.name}_tts" / f"keypoints_{a.name}"
    out.mkdir(parents=True, exist_ok=True)
    rows, allev = [], []
    for tr in test:
        g_path = Path(a.gen_dir) / f"{tr['key']}.wav"
        if not g_path.exists():
            continue
        r, g = tracks(clip_path(tr["id"])), tracks(g_path)
        _, wp = librosa.sequence.dtw(X=r["mf"], Y=g["mf"], metric="cosine")
        wp = wp[::-1]
        r2g = {}
        g2r = {}
        for i, j in wp:
            r2g.setdefault(int(i), int(j))
            g2r.setdefault(int(j), int(i))
        warp = lambda i: r2g.get(int(i), int(i))  # noqa: E731
        warp_inv = lambda i: r2g.get(int(i), -1)  # noqa: E731
        segs, _ = asr.transcribe(str(clip_path(tr["id"])), language="ja", word_timestamps=True, beam_size=5,
                                 condition_on_previous_text=False)
        words = [{"word": w.word, "start": w.start, "end": w.end} for s in segs for w in (s.words or [])]
        r_ev, g_ev = events(r), events(g)
        n = len(r["st"])
        missed = []
        for e in r_ev:
            ok = kept(e, g_ev, warp)
            allev.append({"key": tr["key"], "kind": e[0], "size": round(e[2], 1), "kept": ok, "place": place(e[1], n),
                          "word": word_at(words, e[1] * STEP).strip(), "emotion": tr["emotion"], "type": tr["type"]})
            if not ok:
                missed.append(e)
        lr, lg = lively(r), lively(g)
        rows.append({"key": tr["key"], "text": tr["text"], "n_ev": len(r_ev), "missed": len(missed),
                     **{f"real_{k}": round(v, 2) for k, v in lr.items()}, **{f"gen_{k}": round(v, 2) for k, v in lg.items()}})
        import soundfile as sf

        for tag, src in (("real", clip_path(tr["id"])), ("gen", g_path)):
            x, sr = read_mono(src)
            sf.write(out / f"{tr['key']}_{tag}.wav", x / (np.abs(x).max() + 1e-9) * 0.8, sr, subtype="PCM_16")
        plot(out / f"{tr['key']}.png", r, g, warp_inv, missed, words,
             f"{tr['key']}  {tr['text']}   关键点 {len(r_ev)}，没做出 {len(missed)}")
    import collections

    tot = len(allev)
    k_rate = np.mean([e["kept"] for e in allev])
    print(f"{len(rows)} lines, {tot} key points in the real clips ({tot / len(rows):.1f} per line); kept by '{a.name}': {k_rate * 100:.0f}%")
    for f, vals in (("kind", ["peak", "jump", "final", "accent"]), ("place", ["start", "middle", "end"])):
        print(f"  by {f}: " + ", ".join(f"{v} {np.mean([e['kept'] for e in allev if e[f] == v]) * 100:.0f}% of "
                                         f"{sum(e[f] == v for e in allev)}" for v in vals if any(e[f] == v for e in allev)))
    for kind in ("final",):
        ups = [e for e in allev if e["kind"] == kind and e["size"] > 0]
        dn = [e for e in allev if e["kind"] == kind and e["size"] < 0]
        print(f"  final rises kept {np.mean([e['kept'] for e in ups]) * 100:.0f}% of {len(ups)}, final falls {np.mean([e['kept'] for e in dn]) * 100:.0f}% of {len(dn)}")
    jumps_up = [e for e in allev if e["kind"] == "jump" and e["size"] > 0]
    jumps_dn = [e for e in allev if e["kind"] == "jump" and e["size"] < 0]
    print(f"  jumps up kept {np.mean([e['kept'] for e in jumps_up]) * 100:.0f}% of {len(jumps_up)}, down {np.mean([e['kept'] for e in jumps_dn]) * 100:.0f}% of {len(jumps_dn)}")
    print("  most missed words:", collections.Counter(e["word"] for e in allev if not e["kept"] and e["word"]).most_common(15))
    for k in ("spread", "speed", "peaks", "loud"):
        ratio = [r_[f"gen_{k}"] / r_[f"real_{k}"] for r_ in rows if r_[f"real_{k}"] and np.isfinite(r_[f"real_{k}"]) and np.isfinite(r_[f"gen_{k}"])]
        print(f"  liveliness {k:7}: generated / real median {np.median(ratio):.2f}; below 0.7 in {np.mean(np.array(ratio) < 0.7) * 100:.0f}% of lines")
    flat = sorted(rows, key=lambda x: (x["gen_spread"] / x["real_spread"]) if x["real_spread"] else 9)[:8]
    print("  flattest lines:", [(x["key"], round(x["gen_spread"] / x["real_spread"], 2), x["text"][:16]) for x in flat])
    json.dump({"rows": rows, "events": allev}, open(out / "keypoints.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    html = ["<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
            "<title>语调关键点</title><style>:root{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--line:#e7e5e4;--card:#fff}",
            "@media (prefers-color-scheme: dark){:root{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--line:#44403c;--card:#292524}}",
            "body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,'Microsoft YaHei',sans-serif}",
            "main{max-width:980px;margin:0 auto;padding:12px 16px 60px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:12px 0}",
            "img{width:100%;height:auto;border-radius:4px;background:#fff}.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}audio{width:100%;height:32px}",
            ".hint{color:var(--muted);font-size:13px}@media (max-width:640px){.row{grid-template-columns:1fr}}</style></head><body><main>",
            f"<h1 style='font-size:18px'>语调关键点：{voice_cfg()['label']}原声 vs {a.name}</h1>",
            "<p class='hint'>每句一张图：蓝线是原声的音高，橙线是生成的音高（按发音对齐到原声的时间轴），红线是原声里有、生成没做出来的关键点（峰：音高峰；跳：大幅升降；尾：句尾上扬或下沉；重：音量突起）。下方是逐词时间。按“漏掉的关键点数”从多到少排列。</p>"]
    for r_ in sorted(rows, key=lambda x: -x["missed"]):
        k = r_["key"]
        html.append(f"<div class='card'><b>{k}</b> {r_['text']} <span class='hint'>关键点 {r_['n_ev']}，没做出 {r_['missed']}；音高起伏 原声 {r_['real_spread']} / 生成 {r_['gen_spread']} 半音</span>"
                    f"<img src='{k}.png' alt='{k} 音高曲线'><div class='row'><div><span class='hint'>原声</span><audio controls preload='none' src='{k}_real.wav'></audio></div>"
                    f"<div><span class='hint'>生成</span><audio controls preload='none' src='{k}_gen.wav'></audio></div></div></div>")
    html.append("</main></body></html>")
    (out / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(f"page: {out / 'index.html'}")
