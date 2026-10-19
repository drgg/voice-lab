"""voice-lab's window: a local page over the launcher and the live tool (tools env, standard library only).

  envs/tools/Scripts/pythonw.exe scripts/ui_server.py        what 一键启动.cmd runs: serve, then open the window
  envs/tools/Scripts/python.exe scripts/ui_server.py --no-open [--port 8766]     serve only (previews, tests)
The window is an Edge app window (no address bar) when Edge is installed, else a browser tab. Everything stays on
this machine: the server listens on 127.0.0.1 only, answers its own page only (the Host header; every action
carries this run's random token in a request header, which other sites can neither read nor send), and quits
once the window has been gone for 10 minutes with nothing running.
Started again while it runs, it only opens another window on the running one.

Every action is a job: launch.ps1 with switches (the same code as the text menu, 一键启动.cmd -Menu), its output
streamed to the page. One job at a time; the live tool is a job too, stopped through its STOP file (a clean exit,
recordings kept). Errors of the server itself go to logs/ui_server.log.
"""
import argparse
import codecs
import ctypes
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import app_route as AR  # noqa: E402
from vl_paths import _VOICE_DEFAULTS, FIRST_VOICE, ROOT, char_paths  # noqa: E402

LAUNCH = ROOT / "scripts" / "launch.ps1"
UI = ROOT / "scripts" / "ui" / "index.html"
TOOLS_PY = ROOT / "envs" / "tools" / "Scripts" / "python.exe"
CACHE = Path(os.environ.get("VL_CACHE") or r"C:\voice-lab-cache")
LIVE_LOGS = ROOT / "logs" / "live"
SETTINGS = ROOT / "logs" / "launcher.json"       # the chosen voice, shared with launch.ps1
LAST_CHECK = ROOT / "logs" / "ui_last_check.json"
SLOG = ROOT / "logs" / "ui_server.log"
PAGE_PORT = 8765                                 # the listening pages' server (python -m http.server)
IDLE_QUIT = 600
NO_WINDOW, DETACHED, NEW_GROUP = 0x08000000, 0x00000008, 0x00000200
ENVS = {  # as launch.ps1's $PYENVS: where each Python environment lives, and whether the pipeline needs it
    "tools": (ROOT / "envs" / "tools", True), "indextts": (CACHE / "envs" / "indextts", True),
    "applio": (CACHE / "envs" / "applio", True), "audiosr": (ROOT / "envs" / "audiosr", True),
    "seedvc": (ROOT / "envs" / "seedvc", True), "vevo2": (ROOT / "envs" / "vevo2", False)}
STATUS_LINE = re.compile(r"^\d+(\.\d+)?s\s+in\s")   # the live tool's running status, rewritten in place
SESSION = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
TOKEN = secrets.token_hex(16)                    # this run's: put into the page, required on every action


def slog(msg):
    SLOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SLOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def clean_arg(s, n=80):
    """A value handed to launch.ps1 as one argument: never read as a switch (no leading '-'), no quotes."""
    return str(s or "").replace('"', "").replace("'", "").strip().lstrip("-")[:n]


# ---------------------------------------------------------------- what the page shows
def voices():
    """data/<voice> with voice.json or finished lines (exp/all[_<voice>]/final_48k), as launch.ps1's Get-Voices."""
    out = []
    for d in sorted(p for p in (ROOT / "data").iterdir() if p.is_dir()):
        fin = ROOT / "exp" / ("all" if d.name == FIRST_VOICE else f"all_{d.name}") / "final_48k"
        cfg_p = d / "voice.json"
        if not cfg_p.exists() and not fin.exists():
            continue
        cfg = dict(_VOICE_DEFAULTS)
        if cfg_p.exists():
            cfg.update(json.loads(cfg_p.read_text(encoding="utf-8-sig")))
        out.append({"name": d.name, "label": cfg["label"], "model": cfg.get("rvc_model") or f"{d.name}_v1",
                    "lines": len(list(fin.glob("*.wav"))) if fin.exists() else 0})
    return out


def current_voice(vs):
    names = [v["name"] for v in vs]
    try:
        v = json.loads(SETTINGS.read_text(encoding="utf-8-sig"))["voice"]
        if v in names:
            return v
    except (OSError, ValueError, KeyError, TypeError):
        pass
    m = re.search(r"\[string\]\$Voice\s*=\s*'([^']+)'", (ROOT / "scripts" / "live.ps1").read_text(encoding="utf-8-sig"))
    return m.group(1) if m and m.group(1) in names else (names[0] if names else "")


def sources():
    """Where environments can be restored from, in launch.ps1's order (its -From takes the name)."""
    snaps = []
    for d in sorted((ROOT / "env-snapshots").glob("*"), reverse=True):
        if d.is_dir():
            try:
                m = json.loads((d / "snapshot.json").read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                m = {}
            snaps.append({"name": d.name, "kind": "快照", "note": m.get("label", ""), "auto": bool(m.get("auto"))})
    out = [s for s in snaps if not s["auto"]]
    for d in sorted((ROOT / "release").glob("repro-*"), reverse=True):
        if (d / "env" / "requirements-tools.txt").exists():
            out.append({"name": d.name, "kind": "复现包", "note": "", "auto": False})
    for d in (ROOT / "requirements", ROOT / "release_tools" / "public" / "requirements"):
        if (d / "requirements-tools.txt").exists():
            out.append({"name": "requirements", "kind": "依赖清单", "note": "随代码附带", "auto": False})
            break
    return out + [s for s in snaps if s["auto"]]


MAIN_CHAR = "source"                             # the live tool's own character (VL_CHAR unset); the rest go in --also
GROUP_LABELS = {"final_48k": "IndexTTS + RVC", "final_48k_v1": "IndexTTS + RVC（旧版 v1）",
                "final_gsv_a_48k": "GPT-SoVITS · A", "final_gsv_keyed_48k": "GPT-SoVITS · 关键点"}
ALSO_DEFAULT = {}   # as live.ps1 without a lineup


def _json(p, default):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def cast():
    """Per game character (data/<char>/sets/all): the voices and versions with finished lines for it (exp/<char's
    set>[_<voice>]/final*/final.csv), the one picked (logs/launcher.json "lineup", which live.ps1 turns into --final
    and --also; else the live tool's defaults; "" = not replaced) and whether it is in the party (party.json)."""
    settings, off = _json(SETTINGS, {}), set(_json(LIVE_LOGS / "party.json", {}).get("off", []))
    lineup = settings.get("lineup") or {}
    vdirs = [d for d in sorted((ROOT / "data").iterdir()) if (d / "voice.json").exists() or d.name == FIRST_VOICE]
    chars = [d.name for d in sorted((ROOT / "data").iterdir()) if (d / "sets" / "all" / "lines" / "lines.csv").exists()]
    chars = sorted(chars, key=lambda c: c != MAIN_CHAR)
    out = []
    for c in chars:
        label = _json(ROOT / "data" / c / "char.json", {}).get("label") or ("原角色" if c == MAIN_CHAR else c)
        opts = []
        for d in vdirs:
            cfg = {**_VOICE_DEFAULTS, **_json(d / "voice.json", {})}
            exp = char_paths(c, "all", d.name)["exp"]
            for g in sorted(exp.glob("final*")):
                if (g / "final.csv").exists():
                    opts.append({"voice": d.name, "voice_label": cfg["label"], "group": g.name,
                                 "group_label": GROUP_LABELS.get(g.name, g.name), "lines": len(list(g.glob("*.wav")))})
        pick = lineup.get(c)
        if pick is None:
            v, g = ((current_voice(voices()), "final_48k") if c == MAIN_CHAR else ALSO_DEFAULT.get(c, ("", "")))
            pick = {"voice": v, "group": g}
        if pick.get("voice") and not any(o["voice"] == pick["voice"] and o["group"] == pick.get("group") for o in opts):
            pick = {"voice": opts[0]["voice"], "group": opts[0]["group"]} if c == MAIN_CHAR and opts else {"voice": ""}
        out.append({"id": c, "label": label, "main": c == MAIN_CHAR, "options": opts,
                    "pick": pick if pick.get("voice") else None, "on": c not in off})
    return out


def party():
    """The characters the live tool replaces, in its order (the main one, then each picked --also: Ctrl+Alt+<n>),
    and which are switched off (logs/live/party.json, shared with the tool; read by it while it runs)."""
    cs = [c for c in cast() if c["pick"]]
    return [{"id": c["id"], "label": c["label"], "on": c["on"], "key": f"Ctrl+Alt+{i + 1}"} for i, c in enumerate(cs)]


def last_session():
    for d in sorted((p for p in LIVE_LOGS.glob("*") if p.is_dir() and SESSION.match(p.name)), reverse=True):
        try:
            s = json.loads((d / "session.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        review = d / "review" / "review.html"
        return {"name": d.name, "seconds": s.get("seconds", 0), "replaced": s.get("counts", {}).get("accept", 0),
                "marks": s.get("marks", 0), "voices": s.get("voices", 0), "underruns": s.get("underruns", 0),
                "review": review.relative_to(ROOT).as_posix() if review.exists() else ""}
    return None


class _Mem(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
               [(n, ctypes.c_ulonglong) for n in ("total", "avail", "tpf", "apf", "tv", "av", "aev")]


def system():
    m = _Mem()
    m.dwLength = ctypes.sizeof(_Mem)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    disks = []
    for drv in dict.fromkeys(("C:", ROOT.drive, CACHE.drive)):
        try:
            disks.append({"drive": drv[0], "free_gb": round(shutil.disk_usage(drv + "\\").free / 2**30, 1)})
        except OSError:
            pass
    return {"ram_total_gb": round(m.total / 2**30, 1), "ram_free_gb": round(m.avail / 2**30, 1), "disks": disks}


def status():
    procs = AR.processes()
    st = AR.load()
    vs = voices()
    return {
        "root": str(ROOT), "voices": vs, "voice": current_voice(vs), "page_server": port_open(PAGE_PORT),
        "envs": [{"name": n, "ok": (d / "Scripts" / "python.exe").exists(), "need": need} for n, (d, need) in ENVS.items()],
        "route": {"game": AR.GAME, "game_running": bool(AR.game_pids(AR.GAME, procs)),
                  "active": bool(st.get("active")), "pending": bool(st.get("pending")),
                  "to": st.get("to_name", ""), "back": st.get("back_name", ""),
                  "watcher": AR._alive(st.get("watcher_pid"), procs), "startup_item": AR.LINK.exists()},
        "live": AR.live_running(procs), "last_session": last_session(), "cast": cast(), "party": party(),
        "sources": sources(),
        "system": system(), "check": APP.check_summary(), "job": APP.job.brief() if APP.job else None,
    }


def ensure_pages():
    """The listening pages' server, as launch.ps1 starts it: python -m http.server on 127.0.0.1:8765."""
    if port_open(PAGE_PORT):
        return True
    subprocess.Popen([str(TOOLS_PY), "-m", "http.server", str(PAGE_PORT), "--bind", "127.0.0.1"], cwd=ROOT,
                     creationflags=DETACHED | NEW_GROUP, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
    for _ in range(40):
        time.sleep(0.25)
        if port_open(PAGE_PORT):
            return True
    return False


def pages():
    up = ensure_pages()
    files = sorted((ROOT / "eval").rglob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)[:14]
    reviews = sorted((p / "review" / "review.html" for p in LIVE_LOGS.glob("*") if SESSION.match(p.name)),
                     key=lambda p: p.parent.parent.name, reverse=True)
    out = []
    for f, kind in [(f, "eval") for f in files] + [(r, "live") for r in reviews if r.exists()][:3]:
        head = f.read_text(encoding="utf-8", errors="replace")[:4000]
        t = re.search(r"<title>([^<]+)</title>", head)
        out.append({"rel": f.relative_to(ROOT).as_posix(), "title": t.group(1).strip() if t else f.stem,
                    "time": time.strftime("%m-%d %H:%M", time.localtime(f.stat().st_mtime)), "kind": kind})
    return {"server": up, "base": f"http://localhost:{PAGE_PORT}/", "pages": out}


# ---------------------------------------------------------------- jobs
class Job:
    """launch.ps1 with some switches; its output kept for the page (lines rewritten in place become `status`)."""
    seq = 0

    def __init__(self, kind, title, args, stoppable=False, on_done=None):
        Job.seq += 1
        self.id, self.kind, self.title, self.args, self.stoppable = Job.seq, kind, title, args, stoppable
        self.on_done, self.lines, self.status, self.exit, self.result = on_done, [], "", None, None
        self.started, self.ended = time.time(), None
        env = {k: v for k, v in os.environ.items() if k.upper() not in ("FORCE_COLOR", "CLICOLOR_FORCE")}
        env.update(PYTHONUNBUFFERED="1", NO_COLOR="1")   # plain text: the page parses some of it
        self.proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(LAUNCH), *args], cwd=ROOT,
            env=env, creationflags=NO_WINDOW, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        threading.Thread(target=self._read, daemon=True).start()

    def _line(self, s):
        s = s.rstrip()
        if STATUS_LINE.match(s.strip()):
            self.status = s.strip()
        elif s.strip() or (self.lines and self.lines[-1]):
            self.lines.append(s)

    def _read(self):
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        buf, cr = "", False
        fd = self.proc.stdout.fileno()
        while True:
            b = os.read(fd, 4096)
            if not b:
                break
            for ch in dec.decode(b):
                if cr and ch != "\n":                  # a lone \r: what came before it was a progress line
                    if buf.strip():
                        self.status = buf.strip()
                    buf = ""
                cr = ch == "\r"
                if ch == "\n":
                    self._line(buf)
                    buf = ""
                elif not cr:
                    buf += ch
        if buf.strip():
            self._line(buf)
        self.exit = self.proc.wait()
        self.ended = time.time()
        if self.on_done:
            try:
                self.on_done(self)
            except Exception as e:  # noqa: BLE001
                self.lines.append(f"(读取结果失败：{e!r})")

    @property
    def running(self):
        return self.exit is None

    def brief(self):
        return {"id": self.id, "kind": self.kind, "title": self.title, "running": self.running, "exit": self.exit,
                "status": self.status, "n": len(self.lines), "started": self.started, "ended": self.ended,
                "stoppable": self.stoppable}

    def view(self, since):
        return {**self.brief(), "lines": self.lines[since:], "result": self.result}

    def stop(self):
        if self.kind == "live":
            (LIVE_LOGS / "STOP").touch()                # the tool's own clean exit
        elif self.stoppable and self.running:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], creationflags=NO_WINDOW,
                           capture_output=True)


class App:
    def __init__(self):
        self.job, self.lock, self.seen = None, threading.Lock(), time.time()
        try:
            self.check = json.loads(LAST_CHECK.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.check = None

    def check_summary(self):
        if not self.check:
            return None
        items = self.check["items"]
        count = {k: sum(1 for i in items if i["Status"] == k) for k in ("OK", "WARN", "FAIL", "INFO", "OPT")}
        return {"time": self.check["time"], "deep": self.check["deep"], "count": count,
                "fixable": sum(1 for i in items if i.get("Fix") and i["Status"] in ("FAIL", "WARN"))}

    def start(self, p):
        """A job from the page's request; only these kinds, with their arguments checked."""
        kind = p.get("kind")
        vs = voices()
        if kind == "live":
            v = p.get("voice") or current_voice(vs)
            if v not in [x["name"] for x in vs]:
                raise ValueError(f"没有声线 {v}")
            args = ["-Live", "-Yes", "-Voice", v] + ([] if p.get("route", True) else ["-NoRoute"])
            if clean_arg(p.get("play")):
                args += ["-Play", clean_arg(p.get("play"))]
            return self._run("live", "实时替换", args, stoppable=True)
        if kind == "check":
            out = ROOT / "cache" / "tmp" / f"ui_check_{time.time_ns()}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            deep = bool(p.get("deep"))
            return self._run("check", "深度检查" if deep else "环境检查",
                             ["-Check", "-JsonOut", str(out)] + (["-Deep"] if deep else []), stoppable=True,
                             on_done=lambda j: self._checked(j, out, deep))
        if kind == "setup":
            return self._run("setup", "环境配置与修复", ["-Setup", "-Yes"] + (["-All"] if p.get("all") else []))
        if kind == "snapshot":
            return self._run("snapshot", "保存环境快照", ["-Snapshot", "-Label", clean_arg(p.get("label")) or "界面保存"])
        if kind in ("restore_preview", "restore"):
            src = p.get("from")
            if src not in [s["name"] for s in sources()]:
                raise ValueError(f"没有恢复来源 {src}")
            if kind == "restore_preview":
                return self._run(kind, "恢复预览", ["-Restore", "-From", src, "-DryRun"], stoppable=True)
            return self._run(kind, "恢复环境", ["-Restore", "-From", src, "-Yes"])
        if kind == "game_default":
            return self._run(kind, "游戏声音切回默认", ["-GameDefault"])
        if kind == "shortcut":
            return self._run(kind, "创建桌面快捷方式", ["-Shortcut"])
        raise ValueError(f"不认识的操作 {kind}")

    def _run(self, kind, title, args, stoppable=False, on_done=None):
        with self.lock:
            if self.job and self.job.running:
                raise RuntimeError(f"“{self.job.title}”还在运行，先等它结束")
            if kind == "live" and AR.live_running():
                raise RuntimeError("实时工具已经在运行（从别处启动的）：先停止它")
            self.job = Job(kind, title, args, stoppable, on_done)
            slog(f"job {self.job.id} {kind}: {' '.join(args)}")
            return self.job.brief()

    def _checked(self, job, out, deep):
        items = json.loads(out.read_text(encoding="utf-8-sig")) if out.exists() else []
        out.unlink(missing_ok=True)
        if isinstance(items, dict):                    # one item comes back as an object
            items = [items]
        job.result = items
        if items:
            self.check = {"time": time.strftime("%Y-%m-%d %H:%M"), "deep": deep, "items": items}
            LAST_CHECK.write_text(json.dumps(self.check, ensure_ascii=False), encoding="utf-8")


APP = App()


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    port = 8766

    def log_message(self, *a):
        pass

    def _own(self):
        return self.headers.get("Host", "") in (f"127.0.0.1:{self.port}", f"localhost:{self.port}")

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self._own():
            return self._send(403, {"error": "forbidden"})
        APP.seen = time.time()
        u = urlparse(self.path)
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, UI.read_text(encoding="utf-8").replace("__VL_TOKEN__", TOKEN).encode("utf-8"),
                                  "text/html; charset=utf-8")
            if u.path == "/api/ping":
                return self._send(200, {"app": "voice-lab"})
            if u.path == "/api/status":
                return self._send(200, status())
            if u.path == "/api/pages":
                return self._send(200, pages())
            if u.path == "/api/check":
                return self._send(200, APP.check)
            if u.path == "/api/job":
                since = int(parse_qs(u.query).get("since", ["0"])[0])
                return self._send(200, APP.job.view(since) if APP.job else None)
            return self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            slog(f"GET {u.path}: {traceback.format_exc()}")
            return self._send(500, {"error": repr(e)})

    def do_POST(self):
        # other sites' pages can neither read the token nor send the header without a preflight (never granted)
        if not self._own() or not secrets.compare_digest(self.headers.get("X-VL", ""), TOKEN):
            return self._send(403, {"error": "forbidden"})
        APP.seen = time.time()
        path = urlparse(self.path).path
        try:
            p = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if path == "/api/job":
                return self._send(200, APP.start(p))
            if path == "/api/stop":                     # the running job, or a live tool started elsewhere
                if APP.job and APP.job.running:
                    APP.job.stop()
                elif AR.live_running():
                    (LIVE_LOGS / "STOP").touch()
                return self._send(200, {"ok": True})
            if path == "/api/voice":
                if p.get("voice") not in [v["name"] for v in voices()]:
                    raise ValueError("没有这个声线")
                SETTINGS.parent.mkdir(parents=True, exist_ok=True)
                SETTINGS.write_text(json.dumps({"voice": p["voice"]}), encoding="utf-8")
                return self._send(200, {"ok": True})
            if path == "/api/cast":                     # a character's voice and version ("" voice: not replaced)
                cur = {c["id"]: c for c in cast()}
                c, v, g = p.get("char"), p.get("voice") or "", p.get("group") or ""
                if c not in cur:
                    raise ValueError("没有这个角色")
                if v and not any(o["voice"] == v and o["group"] == g for o in cur[c]["options"]):
                    raise ValueError("这个角色没有这个版本的成品")
                if not v and cur[c]["main"]:
                    raise ValueError("主角色必须选一个声线")
                st = _json(SETTINGS, {})
                lineup = {k: (x["pick"] or {"voice": ""}) for k, x in cur.items()}  # every character, in order
                lineup[c] = {"voice": v, "group": g} if v else {"voice": ""}
                st["lineup"] = lineup
                if cur[c]["main"]:
                    st["voice"] = v                     # launch.ps1's -Voice and the start button
                SETTINGS.parent.mkdir(parents=True, exist_ok=True)
                SETTINGS.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
                return self._send(200, {"ok": True, "cast": cast(), "party": party(), "voice": st.get("voice")})
            if path == "/api/party":                    # a character in or out of the party (the live tool follows)
                cur = party()
                if p.get("char") not in [c["id"] for c in cur]:
                    raise ValueError("没有这个角色")
                off = {c["id"] for c in cur if not c["on"]}
                (off.discard if p.get("on") else off.add)(p["char"])
                LIVE_LOGS.mkdir(parents=True, exist_ok=True)
                (LIVE_LOGS / "party.json").write_text(json.dumps({"off": sorted(off)}, ensure_ascii=False), encoding="utf-8")
                return self._send(200, {"ok": True, "party": party()})
            if path == "/api/menu":                     # the text launcher in a console of its own
                subprocess.Popen(["cmd", "/c", "start", "voice-lab", str(ROOT / "一键启动.cmd"), "-Menu"],
                                 creationflags=NO_WINDOW)
                return self._send(200, {"ok": True})
            if path == "/api/folder":
                os.startfile(ROOT)
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except RuntimeError as e:
            return self._send(409, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            slog(f"POST {path}: {traceback.format_exc()}")
            return self._send(500, {"error": repr(e)})


class Server(ThreadingHTTPServer):
    allow_reuse_address = False                        # a second server must fail, not share the port
    daemon_threads = True


def open_window(url):
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
        edge = Path(base or ".") / "Microsoft" / "Edge" / "Application" / "msedge.exe"
        if base and edge.exists():
            subprocess.Popen([str(edge), f"--app={url}", "--window-size=1280,900"])
            return
    webbrowser.open(url)


def idle_quit(srv):
    while True:
        time.sleep(15)
        if not (APP.job and APP.job.running) and time.time() - APP.seen > IDLE_QUIT:
            slog("no window for 10 minutes: quitting")
            srv.shutdown()
            return


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-open", action="store_true", help="serve only, open no window")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}/"
    Handler.port = a.port
    try:
        srv = Server(("127.0.0.1", a.port), Handler)
    except OSError:                                    # already running: a window on that one
        if not a.no_open:
            open_window(url)
        return
    slog(f"serving {url}")
    threading.Thread(target=idle_quit, args=(srv,), daemon=True).start()
    if not a.no_open:
        open_window(url)
    srv.serve_forever()


if __name__ == "__main__":
    main()
