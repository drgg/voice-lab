"""One page to compare the finished lines of several target voices side by side (tools env).

For every line of the set: the original, then each voice's final_48k version, with the line's text, route and
where each version came from (final.csv). Audio loads only when played.

  envs/tools/Scripts/python scripts/compare_voices.py --voices default   (VL_SET=all)
  -> eval/<set>/compare_voices.html (open through the page server)
"""
import argparse
import html
import json
import os
from pathlib import Path

from vl_common import LINES, ROOT, display_text, load_lines, read_csv, route_info
from vl_paths import SET_NAME, char_label, page_path, routes

ROUTE = {"speech": "台词", "refine": "台词（需精修）", "nonverbal": "拟声词"}


def voice_label(v):
    p = ROOT / "data" / v / "voice.json"
    return json.loads(p.read_text(encoding="utf-8")).get("label", v) if p.exists() else v


def final_dir(v):
    """exp/<set>/final_48k for the first voice, exp/<set>_<voice>/final_48k for the others (vl_paths.EXP)."""
    s = os.environ.get("VL_SET", "").strip()
    if v == "default":
        return (ROOT / "exp" / s if s else ROOT / "exp") / "final_48k"
    return ROOT / "exp" / f"{SET_NAME}_{v}" / "final_48k"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default="default")
    ap.add_argument("--columns", default="",
                    help='"label=voice:group,..." instead of --voices, e.g. a before/after pair of one voice')
    ap.add_argument("--only-changed", action="store_true", help="only lines whose files differ between the columns")
    ap.add_argument("--title", default="声线对比")
    ap.add_argument("--out", default="compare_voices.html")
    a = ap.parse_args()
    out_dir = (ROOT / "eval" / os.environ["VL_SET"]) if os.environ.get("VL_SET") else ROOT / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.columns:
        cols = []
        for c in a.columns.split(","):
            lab, spec = c.split("=", 1)
            v, g = spec.split(":", 1)
            cols.append((lab, final_dir(v).parent / g))
    else:
        cols = [(voice_label(v), final_dir(v)) for v in a.voices.split(",") if final_dir(v).exists()]
    src = [{r["id"]: r["source"] for r in read_csv(d / "final.csv")} if (d / "final.csv").exists() else {}
           for _, d in cols]
    rt, info = routes(), route_info()
    rows = []
    for ln in load_lines():
        files = [d / f"{ln['id']}.wav" for _, d in cols]
        if a.only_changed and len({f.read_bytes() if f.exists() else b"" for f in files}) < 2:
            continue
        cells = [f'<td><audio controls preload="none" src="{page_path(LINES / (ln["id"] + ".wav"), out_dir)}"></audio></td>']
        for f, s in zip(files, src):
            how = html.escape(s.get(ln["id"], "").replace("_48k", ""))
            cells.append(f'<td><audio controls preload="none" src="{page_path(f, out_dir)}"></audio>'
                         f'<div class="src">{how}</div></td>' if f.exists() else "<td>-</td>")
        rows.append(f'<tr data-route="{rt.get(ln["id"], "speech")}"><td class="id">{ln["id"]}<div class="src">'
                    f'{ROUTE.get(rt.get(ln["id"], "speech"), "")}</div></td><td class="text">{html.escape(display_text(ln, info.get(ln["id"])))}</td>'
                    + "".join(cells) + "</tr>")
    heads = "".join(f"<th>{html.escape(lab)}</th>" for lab, _ in cols)
    voices = [lab for lab, _ in cols]
    page = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(a.title)}</title><style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--line:#e5e5ea;--acc:#0a84ff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1c1c1e;--fg:#f2f2f7;--mut:#a1a1a6;--line:#3a3a3c}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}}
h1{{font-size:20px;margin:0 0 6px}} p.hint{{color:var(--mut);margin:0 0 12px}}
.bar{{margin:8px 0 12px;display:flex;gap:8px;flex-wrap:wrap}} .bar button{{border:1px solid var(--line);background:none;color:var(--fg);
border-radius:6px;padding:4px 10px;cursor:pointer}} .bar button.on{{border-color:var(--acc);color:var(--acc)}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%}} th,td{{border-bottom:1px solid var(--line);padding:6px;vertical-align:top;text-align:left}}
th{{position:sticky;top:0;background:var(--bg)}} audio{{width:210px;height:32px}} .src{{color:var(--mut);font-size:12px}}
.id{{white-space:nowrap}} .text{{min-width:160px;max-width:320px}}
</style></head><body>
<h1>{html.escape(a.title)} · {len(rows)} 句</h1>
<p class="hint">每行：{html.escape(char_label())}原声，然后各声线的最终版（48 kHz）。小字是这一版的来源（重配 + RVC、拟声词 RVC、保留原声、精修…）。音频点播放才加载。</p>
<div class="bar"><button data-f="all" class="on">全部</button><button data-f="speech">台词</button><button data-f="refine">需精修</button><button data-f="nonverbal">拟声词</button></div>
<div class="wrap"><table><thead><tr><th>编号</th><th>台词</th><th>{html.escape(char_label())}原声</th>{heads}</tr></thead><tbody>
{''.join(rows)}
</tbody></table></div>
<script>
document.querySelectorAll('.bar button').forEach(b=>b.onclick=()=>{{document.querySelectorAll('.bar button').forEach(x=>x.classList.toggle('on',x===b));
document.querySelectorAll('tbody tr').forEach(r=>r.style.display=(b.dataset.f==='all'||r.dataset.route===b.dataset.f)?'':'none')}});
document.addEventListener('play',e=>document.querySelectorAll('audio').forEach(x=>{{if(x!==e.target)x.pause()}}),true);
</script></body></html>"""
    (out_dir / a.out).write_text(page, encoding="utf-8")
    print(f"{len(rows)} lines x {len(voices)} voices ({', '.join(voices)}) -> {out_dir / a.out}")
