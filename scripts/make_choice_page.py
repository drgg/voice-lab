"""Pick one version per line for a whole group of lines at once (tools env).

Built for the non-verbal lines: every line shows the original and each candidate version, with the
issues check_nonverbal.py found under each candidate. The listener sets one default version for
all lines - or "auto": per line the first version in --auto order with no issue found - overrides
single lines, and copies a short code such as
    default=auto;L102=orig,L134=tts
which finalize.py --choices applies (keys and auto picks are in exp/<set>/choice_groups.json).

Usage:  envs/tools/Scripts/python scripts/make_choice_page.py --route nonverbal --default rvc \
            --groups "rvc=N_rvc_48k=RVC 新版,orig=N_orig_48k=原声保留,..." --out nonverbal.html
"""
import argparse
import json
from pathlib import Path

from vl_common import EVAL, EXP, LINES, display_text, load_lines, read_csv, route_info
from vl_paths import SET_NAME, page_path, routes

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default="nonverbal")
    ap.add_argument("--groups", required=True, help="key=exp_group=label,... in display order")
    ap.add_argument("--default", required=True, help="key preselected for every line, or auto")
    ap.add_argument("--auto", default="", help="key order for the per-line auto pick, e.g. rvc,rvc2,tts,orig")
    ap.add_argument("--title", default="拟声词统一检查")
    ap.add_argument("--out", default="nonverbal.html")
    a = ap.parse_args()

    groups = [dict(zip(("key", "group", "label"), g.split("=", 2))) for g in a.groups.split(",")]
    issues = {}
    if (EVAL / "nonverbal_check.csv").exists():
        issues = {(r["group"], r["id"]): r["问题"] for r in read_csv(EVAL / "nonverbal_check.csv")}
    route = routes()
    auto_order = [k for k in a.auto.split(",") if k]
    lines, auto = [], {}
    info = route_info()
    for ln in load_lines():
        if route.get(ln["id"]) != a.route:
            continue
        cands = [{"key": g["key"], "src": page_path(EXP / g["group"] / f"{ln['id']}.wav"),
                  "issues": issues.get((g["group"], ln["id"]), "")}
                 for g in groups if (EXP / g["group"] / f"{ln['id']}.wav").exists()]
        # "auto": the first version in --auto order that the check found nothing wrong with, else the last
        have = {c["key"]: c for c in cands}
        ok = [k for k in auto_order if k in have and not have[k]["issues"]]
        auto[ln["id"]] = ok[0] if ok else next((k for k in reversed(auto_order) if k in have), cands[0]["key"])
        lines.append({"id": ln["id"], "text": display_text(ln, info.get(ln["id"])), "src_wav": ln.get("src_wav", ""),
                      "orig": page_path(LINES / f"{ln['id']}.wav"), "cands": cands, "auto": auto[ln["id"]]})
    (EXP / "choice_groups.json").write_text(json.dumps(
        {"groups": {g["key"]: g["group"] for g in groups}, "auto": auto}, indent=1), encoding="utf-8")
    counts = {}
    for k in auto.values():
        counts[k] = counts.get(k, 0) + 1
    print(f"auto picks: {counts}")
    data = {"set": SET_NAME, "title": a.title, "default": a.default,
            "groups": [{"key": g["key"], "label": g["label"]} for g in groups], "lines": lines}
    tpl = (Path(__file__).parent / "templates" / "choose.html").read_text(encoding="utf-8")
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / a.out).write_text(tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False)), encoding="utf-8")
    print(f"{len(lines)} lines x {len(groups)} versions -> {EVAL / a.out}")
