"""Change log for later tuning: every change to the pipeline, its settings or its outputs gets an entry under
logs/changes/ (standard library only).

  python scripts/change_log.py start <slug> --title "..." --why "..." [--files scripts/a.py data/x/voice.json ...]
      new entry logs/changes/<YYYY-MM-DD>_<NN>_<slug>/: README.md (the sections to fill in) and before/<path>
      copies of the files about to change (missing ones are noted as new)
  python scripts/change_log.py diff <entry>
      <entry>/diff.patch: unified diff of before/ against the current files
  python scripts/change_log.py attach <entry> <file>... [--to results]
      copy outputs (metrics, analysis printouts) or one-off scripts into <entry>/<to>/
  python scripts/change_log.py index
      logs/changes/INDEX.md: one line per entry - date, title, status (the README's "状态" line)

<entry> is the folder name or a unique part of it.
"""
import argparse
import datetime
import difflib
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "changes"

TEMPLATE = """# {title}

- 日期：{date}
- 状态：进行中
- 起因：{why}
- 改动的文件：{files}

## 改动内容

## 运行

## 结果（改动前 → 改动后）

## 结论

## 后续可优化

## 回滚

before/ 里是改动前的文件，diff.patch 是改动的差异。
"""


def entry_dir(key):
    hits = [d for d in sorted(LOG.iterdir()) if d.is_dir() and key in d.name]
    if len(hits) != 1:
        sys.exit(f"{len(hits)} entries match {key!r}: {[d.name for d in hits]}")
    return hits[0]


def start(a):
    LOG.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    n = 1 + sum(1 for d in LOG.iterdir() if d.is_dir() and d.name.startswith(today))
    d = LOG / f"{today}_{n:02d}_{a.slug}"
    d.mkdir()
    new = []
    for f in a.files:
        src = (ROOT / f).resolve()
        rel = src.relative_to(ROOT)
        if src.exists():
            dst = d / "before" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            new.append(rel.as_posix())
    (d / "files.txt").write_text("\n".join(Path(f).as_posix() for f in a.files) + "\n", encoding="utf-8")
    files = ", ".join(f"`{Path(f).as_posix()}`" + ("（新文件）" if Path(f).as_posix() in new else "") for f in a.files) or "无"
    (d / "README.md").write_text(TEMPLATE.format(title=a.title, date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                 why=a.why, files=files), encoding="utf-8")
    index(None)
    print(d)


def read_text(p):
    if not p.exists():
        return []
    raw = p.read_bytes()
    return raw.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").splitlines(keepends=True)


def diff(a):
    d = entry_dir(a.entry)
    files = [ln.strip() for ln in (d / "files.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = []
    for f in files:
        old, cur = read_text(d / "before" / f), read_text(ROOT / f)
        out += difflib.unified_diff(old, cur, f"a/{f}", f"b/{f}")
    (d / "diff.patch").write_text("".join(out), encoding="utf-8")
    print(f"{d / 'diff.patch'}: {sum(1 for x in out if x.startswith(('+', '-')) and not x.startswith(('+++', '---')))} changed lines")


def attach(a):
    d = entry_dir(a.entry) / a.to
    d.mkdir(parents=True, exist_ok=True)
    for f in a.paths:
        shutil.copy2(f, d / Path(f).name)
    print(f"{len(a.paths)} files -> {d}")


def index(_a):
    rows = []
    for d in sorted(LOG.iterdir()) if LOG.exists() else []:
        r = d / "README.md"
        if not (d.is_dir() and r.exists()):
            continue
        t = r.read_text(encoding="utf-8")
        title = t.splitlines()[0].lstrip("# ").strip()
        m = re.search(r"状态：(.+)", t)
        rows.append(f"- [{d.name}]({d.name}/README.md) — {title}（{m.group(1).strip() if m else '?'}）")
    (LOG / "INDEX.md").write_text("# 改动日志\n\n每次改动一个条目：起因、改了什么、运行命令、改动前后的指标、结论、回滚方法。\n\n"
                                  + "\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("slug")
    s.add_argument("--title", required=True)
    s.add_argument("--why", required=True)
    s.add_argument("--files", nargs="*", default=[])
    s.set_defaults(fn=start)
    s = sub.add_parser("diff")
    s.add_argument("entry")
    s.set_defaults(fn=diff)
    s = sub.add_parser("attach")
    s.add_argument("entry")
    s.add_argument("paths", nargs="+")
    s.add_argument("--to", default="results")
    s.set_defaults(fn=attach)
    s = sub.add_parser("index")
    s.set_defaults(fn=index)
    a = ap.parse_args()
    a.fn(a)
