"""Build a line set for the production pipeline from all exported lines (tools env).

Takes data/<char>/wav/ + transcript.csv (prep_source.py; VL_CHAR, default source) and writes data/<char>/sets/<name>/lines/
with ids L001, L002, ... (the scripts glob L*.wav) and lines.csv:
  id, src_wav (original file name, for finalize.py's export), duration, bucket, text, text_edit,
  avg_logprob, no_speech_prob (Whisper confidence, read by scan_lines.py)
Re-running keeps the text_edit corrections already made in an existing lines.csv.

Usage:  envs/tools/Scripts/python scripts/prep_set.py --name all
        ... prep_set.py --name smoke --pick <file>.wav,Media__...   (a few lines to test with)
Then:   .\scripts\run_d4.ps1 -Set all
"""
import argparse
import shutil

from vl_common import SOURCE_DIR, read_csv, write_csv


def bucket(d):
    return "short" if d < 2.5 else "medium" if d < 6.0 else "long"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pick", default=None, help="comma list of source wav names (default: every line)")
    a = ap.parse_args()

    rows = sorted(read_csv(SOURCE_DIR / "transcript.csv"), key=lambda r: r["wav"])
    if a.pick:
        wanted = [w.strip() for w in a.pick.split(",")]
        rows = [r for r in rows if r["wav"] in wanted]
        missing = set(wanted) - {r["wav"] for r in rows}
        if missing:
            raise SystemExit(f"not in transcript.csv: {sorted(missing)}")
    lines_dir = SOURCE_DIR / "sets" / a.name / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)
    old = {r["src_wav"]: r for r in read_csv(lines_dir / "lines.csv")} if (lines_dir / "lines.csv").exists() else {}
    out = []
    for k, r in enumerate(rows, 1):
        lid = f"L{k:03d}"
        shutil.copy2(SOURCE_DIR / "wav" / r["wav"], lines_dir / f"{lid}.wav")
        d = float(r["duration"])
        out.append({"id": lid, "src_wav": r["wav"], "duration": round(d, 3), "bucket": bucket(d),
                    "text": r["text"], "text_edit": old.get(r["wav"], {}).get("text_edit", ""),
                    "avg_logprob": r["avg_logprob"], "no_speech_prob": r["no_speech_prob"]})
    write_csv(lines_dir / "lines.csv", out)
    print(f"set '{a.name}': {len(out)} lines -> {lines_dir}")
