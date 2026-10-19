"""Assemble the final version of every line and export it under the original file names (tools env).

Source per line (routes from scan_lines.py):
  refined_48k/<id>.wav    hand-refined lines (run_refine.ps1 -Pick) always win
  V_rvc_48k/<id>.wav      non-verbal lines: RVC on the original performance
  <tts-group>/<id>.wav    everything else: the phrase-level re-dub (default with RVC polish)
Picks from a choice page (make_choice_page.py), e.g. --choices "default=rvc;L102=orig", override the
non-verbal default and single lines; they are kept in exp/<set>/choices.txt and re-applied on later runs.
Writes exp/<set>/final_48k/<id>.wav + final.csv; with --export also out/<set>/<original name>.wav
(48 kHz 16-bit mono, as listed in wem_info.csv) + manifest.csv.

Usage:  envs/tools/Scripts/python scripts/finalize.py [--tts-group D4_indextts_phrase_48k] [--choices ...] [--export]
"""
import argparse
import json
import shutil

from vl_common import EXP, SOURCE_DIR, load_lines, read_csv, write_csv
from vl_paths import OUT, routes


def parse_choices(code):
    """'default=rvc;L102=orig,L134=tts' -> (default group, {line: group}) via exp/<set>/choice_groups.json."""
    cg = json.loads((EXP / "choice_groups.json").read_text(encoding="utf-8"))
    keys, auto = cg["groups"], cg.get("auto", {})
    default, over = None, {}
    for part in filter(None, (s.strip() for s in code.replace("；", ";").split(";"))):
        for kv in filter(None, (s.strip() for s in part.replace("，", ",").split(","))):
            k, v = (s.strip() for s in kv.split("=", 1))
            if k == "default" and v == "auto":  # per-line auto picks made by make_choice_page.py
                over = {**{line: keys[key] for line, key in auto.items()}, **over}
                continue
            if v not in keys:
                raise SystemExit(f"unknown version key {v!r} in {kv!r}; known: {sorted(keys)} or auto")
            if k == "default":
                default = keys[v]
            else:
                over[k] = keys[v]
    return default, over

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts-group", default="D4_indextts_phrase_rvc_48k")
    ap.add_argument("--vc-group", default="V_rvc_48k")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--choices", default=None, help="code from a choice page (stored for later runs)")
    a = ap.parse_args()

    if a.choices:
        (EXP / "choices.txt").write_text(a.choices.strip(), encoding="utf-8")
    nv_default, over = None, {}
    if (EXP / "choices.txt").exists():
        nv_default, over = parse_choices((EXP / "choices.txt").read_text(encoding="utf-8"))
        print(f"choices: non-verbal default {nv_default}, {len(over)} single-line picks")
    route = routes()
    final = EXP / "final_48k"
    final.mkdir(parents=True, exist_ok=True)
    wem = {r["wav"]: r for r in read_csv(SOURCE_DIR / "wem_info.csv")} if (SOURCE_DIR / "wem_info.csv").exists() else {}
    rows, missing = [], []
    for ln in load_lines():
        r = route.get(ln["id"], "speech")
        vc = (nv_default or a.vc_group) if r == "nonverbal" else a.vc_group
        order = ["refined_48k"] + ([vc, a.tts_group] if r == "nonverbal" else [a.tts_group, vc])
        if ln["id"] in over:  # a single-line pick from the choice page beats everything else
            order.insert(0, over[ln["id"]])
        src = next((EXP / g / f"{ln['id']}.wav" for g in order if (EXP / g / f"{ln['id']}.wav").exists()), None)
        if src is None:
            missing.append(ln["id"])
            continue
        shutil.copy2(src, final / f"{ln['id']}.wav")
        info = wem.get(ln.get("src_wav", ""), {})
        rows.append({"id": ln["id"], "src_wav": ln.get("src_wav", ""), "route": r, "source": src.parent.name,
                     "wem": info.get("source", ""), "encoding": info.get("encoding", ""),
                     "sample_rate": info.get("sample_rate", ""), "channels": info.get("channels", ""),
                     "text": ln["ref_text"]})
    write_csv(final / "final.csv", rows)
    by_source = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    print(f"final_48k: {len(rows)} lines {by_source}" + (f", MISSING {missing}" if missing else ""))

    if a.export:
        OUT.mkdir(parents=True, exist_ok=True)
        for r in rows:
            name = r["src_wav"] or f"{r['id']}.wav"
            shutil.copy2(final / f"{r['id']}.wav", OUT / name)
            r["out_file"] = name
        write_csv(OUT / "manifest.csv", rows)
        print(f"exported {len(rows)} files under their original names -> {OUT}")
