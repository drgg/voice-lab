"""Build listening pages (tools env).

Blind (default): every group's output in a shuffled, anonymous order (X1, X2, ...); the label ->
group mapping goes to eval/listening_key.json and is only read back by summarize.py.
Labeled (--labeled): method names shown, fixed order - for quick by-ear comparison of chosen groups.

Usage:  python scripts/make_listening_page.py [--seed 2026]
        python scripts/make_listening_page.py --labeled --methods A_seedvc_timbre,A2_seedvc_emo --out compare.html
"""
import argparse
import json
import random
from pathlib import Path

from vl_common import EVAL, EXP, LINES, METHODS, TARGET, display_text, load_lines, route_info
from vl_paths import char_label, page_path, voice_cfg

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--methods", default=None, help="comma list; default = every group with outputs")
    ap.add_argument("--labeled", action="store_true")
    ap.add_argument("--out", default="listening.html")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    wanted = a.methods.split(",") if a.methods else list(METHODS)
    abx = sorted((TARGET / "refs").glob("abx_*.wav"))
    items, key = [], {}
    info = route_info()
    for i, ln in enumerate(load_lines()):
        cands = [m for m in wanted if (EXP / m / f"{ln['id']}.wav").exists()]
        if a.labeled:
            labels = {METHODS.get(m, m): m for m in cands}
        else:
            rng.shuffle(cands)
            labels = {f"X{k + 1}": m for k, m in enumerate(cands)}
        key[ln["id"]] = labels
        items.append({
            "id": ln["id"], "text": display_text(ln, info.get(ln["id"])),
            "orig": page_path(LINES / f"{ln['id']}.wav"),
            "ref02": page_path(abx[i % len(abx)]) if abx else "",
            "cands": [{"label": lab, "src": page_path(EXP / m / f"{ln['id']}.wav")} for lab, m in labels.items()],
        })
    tpl = (Path(__file__).parent / "templates" / "listening.html").read_text(encoding="utf-8")
    page = tpl.replace("__ITEMS__", json.dumps(items, ensure_ascii=False)).replace("__VOICE__", voice_cfg()["label"]) \
        .replace("__CHAR__", char_label())
    if a.labeled:
        page = page.replace("<title>声线盲听评分</title>", "<title>声线对比试听</title>") \
                   .replace("<h1>声线盲听评分</h1>", "<h1>声线对比试听</h1>") \
                   .replace("'vl_listening_v1'", "'vl_compare_v1'")  # keep notes apart from the blind test
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / a.out).write_text(page, encoding="utf-8")
    if not a.labeled:
        (EVAL / "listening_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"{len(items)} lines, groups per line: {[len(it['cands']) for it in items]} -> {EVAL / a.out}")
