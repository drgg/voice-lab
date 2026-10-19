"""The game's output device, switched automatically: to the virtual device while the live tool runs, back to where
it was when the tool stops (tools env; standard library only, so the background watcher needs nothing else).

Windows remembers an app's output device across launches (Settings > System > Sound > App volume and device
preferences), and it can only be set while the app is running. So:
  - while the live tool runs, its device process sends the game to the virtual device as soon as the game appears;
  - when the tool stops and the game still runs, the game gets its device back at once: the Windows default, or the
    real device you had picked for it yourself;
  - when the game was closed first, a small background watcher waits for its next start without the live tool and
    gives the device back then (it exits right after, or as soon as the live tool runs again). While such a
    switch-back is pending, a shortcut in your Startup folder starts the watcher after a reboot too; it is removed
    as soon as nothing is pending any more.
The setting is the same one the Settings page changes, reached through Windows' undocumented
IAudioPolicyConfigFactory the way EarTrumpet does. The game process itself is never opened: it is found in a
system process snapshot, and the audio service applies the setting.

  python scripts/app_route.py status     JSON: the game's device now, the switch state, watcher, live tool
  python scripts/app_route.py restore    give the game its device back now (or arm the watcher if it is closed)
  python scripts/app_route.py reset      send the game to the Windows default (now, or at its next start)
Every change is logged to logs/live/route.log. Tests: VL_ROUTE_DRY=1 logs what would be set and changes nothing;
VL_ROUTE_DIR puts the state, the log and the startup shortcut elsewhere.
"""
import ctypes
import json
import os
import subprocess
import sys
import time
import uuid
from ctypes import POINTER, WINFUNCTYPE, byref, c_int, c_long, c_uint32, c_void_p, wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE_LOGS = Path(os.environ.get("VL_ROUTE_DIR") or ROOT / "logs" / "live")   # VL_ROUTE_DIR: tests elsewhere
STATE = LIVE_LOGS / "route.json"      # the game's routing while it is the live tool's doing
RLOG = LIVE_LOGS / "route.log"        # every change, one JSON line each
LIVE_PID = LIVE_LOGS / "live.pid"     # written by live_replace.py while it runs
GAME = os.environ.get("VL_GAME_PROCESS", "")
DRY = os.environ.get("VL_ROUTE_DRY", "") not in ("", "0")
DEFAULT = "Windows default"
STARTUP = (LIVE_LOGS / "Startup" if os.environ.get("VL_ROUTE_DIR") else
           Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup")
LINK = STARTUP / "voice-lab 游戏声音切回（临时）.lnk"   # only there while a switch-back is pending

TOKEN, RENDER = r"\\?\SWD#MMDEVAPI#", "#{e6327cad-dcec-4949-ae8a-991e976a79d2}"   # endpoint id -> interface path
IIDS = ("ab3d4648-e242-459f-b02f-541c70306324",   # Windows 10 21H2 and later
        "2a59116d-6c4f-45e0-a74f-707e3fef9258")   # Windows 10 1803 - 21H1 (same method layout)
# vtable slots: IInspectable's 6, then 19 methods not used here, then Set / Get. Slot 27 clears every app's
# setting and is never called.
SET, GET = 25, 26
E_RENDER, MULTIMEDIA, CONSOLE = 0, 1, 0


class GUID(ctypes.Structure):
    _fields_ = [("b", ctypes.c_ubyte * 16)]

    @classmethod
    def of(cls, s):
        g = cls()
        ctypes.memmove(g.b, uuid.UUID(s).bytes_le, 16)
        return g


_cb = ctypes.WinDLL("combase")
_cb.RoInitialize.argtypes, _cb.RoInitialize.restype = [c_int], c_long
_cb.WindowsCreateString.argtypes, _cb.WindowsCreateString.restype = [wintypes.LPCWSTR, c_uint32, POINTER(c_void_p)], c_long
_cb.WindowsDeleteString.argtypes, _cb.WindowsDeleteString.restype = [c_void_p], c_long
_cb.WindowsGetStringRawBuffer.argtypes, _cb.WindowsGetStringRawBuffer.restype = [c_void_p, POINTER(c_uint32)], c_void_p
_cb.RoGetActivationFactory.argtypes = [c_void_p, POINTER(GUID), POINTER(c_void_p)]
_cb.RoGetActivationFactory.restype = c_long


def _hstring(s):
    h = c_void_p()
    if s:
        hr = _cb.WindowsCreateString(s, len(s), byref(h))
        if hr < 0:
            raise OSError(f"WindowsCreateString: 0x{hr & 0xFFFFFFFF:08X}")
    return h


class Policy:
    """Windows' per-app default output device."""

    def __init__(self):
        _cb.RoInitialize(1)   # multithreaded; S_FALSE / RPC_E_CHANGED_MODE when COM is already set up: both fine
        name = _hstring("Windows.Media.Internal.AudioPolicyConfig")
        self.p = c_void_p()
        try:
            for iid in IIDS:   # the one this Windows has; the other fails cleanly (E_NOINTERFACE)
                if _cb.RoGetActivationFactory(name, byref(GUID.of(iid)), byref(self.p)) >= 0 and self.p.value:
                    self.iid = iid
                    break
            else:
                raise OSError("per-app output devices need Windows 10 1803 or later")
        finally:
            _cb.WindowsDeleteString(name)
        vt = ctypes.cast(self.p, POINTER(POINTER(c_void_p))).contents
        self._set = WINFUNCTYPE(c_long, c_void_p, c_uint32, c_int, c_int, c_void_p)(vt[SET])
        self._get = WINFUNCTYPE(c_long, c_void_p, c_uint32, c_int, c_int, POINTER(c_void_p))(vt[GET])
        self._dry = {}                            # VL_ROUTE_DRY: what would have been set, read back by get()

    def get(self, pid):
        """The output device the app is set to (an endpoint id; '' = it follows the Windows default). Fails
        (E_INVALIDARG) for sandboxed helper processes; the process that holds the app's audio answers."""
        if pid in self._dry:
            return self._dry[pid]
        h = c_void_p()
        hr = self._get(self.p, pid, E_RENDER, MULTIMEDIA, byref(h))
        if hr < 0:
            raise OSError(f"reading the output device of process {pid}: 0x{hr & 0xFFFFFFFF:08X}")
        if not h.value:
            return ""
        try:
            n = c_uint32()
            s = ctypes.wstring_at(_cb.WindowsGetStringRawBuffer(h, byref(n)), n.value)
        finally:
            _cb.WindowsDeleteString(h)
        return s.removeprefix(TOKEN).removesuffix(RENDER)

    def set(self, pid, dev_id):
        """Send the app's sound to dev_id ('' = back to the Windows default); Windows keeps it for that app."""
        if DRY:
            if pid not in self._dry:
                self.get(pid)                     # refuses like the real call would (no audio opened yet)
            self._dry[pid] = dev_id
            log_route({"type": "dry_set", "pid": pid, "to": dev_id or DEFAULT})
            return
        h = _hstring(TOKEN + dev_id + RENDER if dev_id else "")
        try:
            for role in (MULTIMEDIA, CONSOLE):
                hr = self._set(self.p, pid, E_RENDER, role, h)
                if hr < 0:
                    raise OSError(f"setting the output device of process {pid}: 0x{hr & 0xFFFFFFFF:08X}")
        finally:
            if h.value:
                _cb.WindowsDeleteString(h)


# ---------------------------------------------------------------- processes (one snapshot, no process handles)
class _Entry(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)]


_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.CreateToolhelp32Snapshot.argtypes, _k32.CreateToolhelp32Snapshot.restype = [wintypes.DWORD, wintypes.DWORD], c_void_p
_k32.Process32FirstW.argtypes = _k32.Process32NextW.argtypes = [c_void_p, POINTER(_Entry)]
_k32.Process32FirstW.restype = _k32.Process32NextW.restype = wintypes.BOOL
_k32.CloseHandle.argtypes = [c_void_p]
_k32.GetCurrentProcess.restype = c_void_p
_k32.SetPriorityClass.argtypes = [c_void_p, wintypes.DWORD]


def processes():
    """{pid: exe name} of everything running."""
    snap = _k32.CreateToolhelp32Snapshot(0x2, 0)                  # TH32CS_SNAPPROCESS
    if not snap or snap == c_void_p(-1).value:
        raise OSError(f"process snapshot failed: {ctypes.get_last_error()}")
    out, e = {}, _Entry()
    e.dwSize = ctypes.sizeof(_Entry)
    try:
        ok = _k32.Process32FirstW(snap, byref(e))
        while ok:
            out[e.th32ProcessID] = e.szExeFile
            ok = _k32.Process32NextW(snap, byref(e))
    finally:
        _k32.CloseHandle(snap)
    return out


def game_pids(process=GAME, procs=None):
    return [p for p, n in (procs or processes()).items() if n.lower() == process.lower()]


def _alive(pid, procs=None):
    return bool(pid) and (procs or processes()).get(int(pid), "").lower() in ("python.exe", "pythonw.exe")


def live_running(procs=None):
    try:
        return _alive(int(LIVE_PID.read_text()), procs)
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------- state and log
def load():
    try:
        return json.loads(STATE.read_text(encoding="utf-8-sig"))   # -sig: also when PowerShell wrote it
    except (OSError, ValueError):
        return {}


def save(st):
    LIVE_LOGS.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def log_route(ev):
    ev = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **ev}
    LIVE_LOGS.mkdir(parents=True, exist_ok=True)
    with open(RLOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return ev


def startup_item(on):
    """The Startup-folder shortcut that runs the watcher after a reboot: there only while a switch-back is pending."""
    try:
        if not on:
            if LINK.exists():
                LINK.unlink()
                log_route({"type": "startup_item", "on": False})
            return
        if LINK.exists():
            return
        STARTUP.mkdir(parents=True, exist_ok=True)
        from comtypes.client import CreateObject   # tools env (the watcher itself only ever removes it)
        tools = ROOT / "envs" / "tools" / "Scripts" / "pythonw.exe"
        lnk = CreateObject("WScript.Shell", dynamic=True).CreateShortcut(str(LINK))
        lnk.TargetPath = str(tools if tools.exists() else Path(sys.executable).with_name("pythonw.exe"))
        lnk.Arguments = f'"{Path(__file__).resolve()}" watch'
        lnk.WorkingDirectory = str(ROOT)
        lnk.Description = "voice-lab: gives the game its output device back at its next start, then removes itself"
        lnk.Save()
        log_route({"type": "startup_item", "on": True, "path": str(LINK)})
    except Exception as e:  # noqa: BLE001  (the watcher still runs until the next reboot)
        log_route({"type": "startup_item_error", "on": on, "error": repr(e)})


# ---------------------------------------------------------------- the live tool's side
class Router:
    """Keeps the game on the virtual device while the live tool runs (called every device round)."""

    def __init__(self, process=GAME):
        self.process, self.policy, self.refused = process, Policy(), set()
        st = load()
        if st.get("pending"):                     # a restore was waiting for the game: the tool owns it again
            st["pending"] = False
            save(st)
            log_route({"type": "pending_taken_over"})
            startup_item(False)

    def ensure(self, target, outs):
        """target, outs: live_devices.Out. Returns the changes made (logged events)."""
        pids = game_pids(self.process)
        if target is None or not pids:
            return []
        changes = []
        for pid in pids:
            try:
                cur = self.policy.get(pid)
            except OSError as e:                  # no audio opened yet (or a helper process): asked again next round
                if pid not in self.refused:
                    self.refused.add(pid)
                    log_route({"type": "route_wait", "pid": pid, "error": str(e)})
                continue
            st = load()
            if not st.get("active"):              # first sight: where it goes without the tool
                real = {o.id: o.name for o in outs if not o.virtual and not o.handsfree}
                st = {"active": True, "pending": False, "process": self.process, "to": target.id,
                      "to_name": target.name, "back_to": cur if cur in real else "",
                      "back_name": real.get(cur, DEFAULT), "since": time.strftime("%Y-%m-%d %H:%M:%S")}
                save(st)
                log_route({"type": "route_start", "pid": pid, "was": cur or DEFAULT, **st})
            if cur != target.id:
                try:
                    self.policy.set(pid, target.id)
                except OSError as e:
                    log_route({"type": "route_error", "pid": pid, "error": str(e)})
                    continue
                changes.append(log_route({"type": "route_set", "pid": pid, "from": cur or DEFAULT, "to": target.name}))
        return changes


def restore(reason="", log=None, quiet=False, defer=True):
    """Give the game its device back: now if it runs, else at its next start (background watcher).
    Windows only takes the setting for a process that has opened audio, so a game that is still loading may refuse
    it for a while: defer hands that to the watcher too ("pending"); without it the answer is "failed" (the
    watcher's own calls). quiet: do not log a refusal.
    Returns ("none" | "restored" | "pending" | "failed", the device name it gets back)."""
    st = load()
    if not st.get("active"):
        return "none", ""
    back = st.get("back_name") or DEFAULT
    pids = game_pids(st.get("process", GAME))
    done, errors = [], {}
    if pids:
        pol = Policy()
        for pid in pids:
            try:
                pol.set(pid, st.get("back_to", ""))
                done.append(pid)
            except OSError as e:                  # no audio opened yet, or a helper process: one is enough
                errors[pid] = str(e)
    if done:
        STATE.unlink(missing_ok=True)
        ev, result = log_route({"type": "route_restored", "pids": done, "to": back, "by": reason}), "restored"
        startup_item(False)
    elif pids and not defer:                      # running, but no process took it (yet): keep the state
        ev, result = {"type": "route_refused", "pids": pids, "by": reason, "errors": errors}, "failed"
        if not quiet:
            log_route(ev)
    else:
        st["pending"] = True
        save(st)
        ev, result = log_route({"type": "route_pending", "to": back, "by": reason}), "pending"
        start_watcher()
        startup_item(True)
    if log:
        log(ev)
    return result, back


def reset(reason="manual"):
    """Send the game to the Windows default, whatever it was set to (now, or at its next start)."""
    st = load()
    st.update({"active": True, "process": st.get("process", GAME), "back_to": "", "back_name": DEFAULT})
    save(st)
    return restore(reason)


def start_watcher():
    st = load()
    if _alive(st.get("watcher_pid")):
        return
    exe = Path(sys.executable)
    w = exe.with_name("pythonw.exe")
    p = subprocess.Popen([str(w if w.exists() else exe), str(Path(__file__).resolve()), "watch"], cwd=str(ROOT),
                         creationflags=0x00000008 | 0x00000200,  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    st = load()
    st["watcher_pid"] = p.pid
    save(st)
    log_route({"type": "watcher_started", "pid": p.pid})


def watch(poll=1.5, patience=600):
    """Background: wait for the game to start without the live tool, give its device back, exit."""
    _k32.SetPriorityClass(_k32.GetCurrentProcess(), 0x4000)    # BELOW_NORMAL_PRIORITY_CLASS
    log_route({"type": "watcher_waiting", "pid": os.getpid()})
    started = None                                # when the game was first seen running
    try:
        while True:
            st, procs = load(), processes()
            if not st.get("active") or not st.get("pending"):
                startup_item(False)
                return log_route({"type": "watcher_done", "why": "nothing pending"})
            if live_running(procs):
                return log_route({"type": "watcher_done", "why": "the live tool runs again"})
            if game_pids(st.get("process", GAME), procs):
                started = started or time.time()
                r, _ = restore("watcher: the game started without the live tool", quiet=True, defer=False)
                if r != "failed":
                    return log_route({"type": "watcher_done", "why": r,
                                      "after_s": round(time.time() - started, 1)})
                if time.time() - started > patience:   # Windows takes it once the game has opened its audio
                    restore("watcher: last try", defer=False)   # logs the refusal
                    return log_route({"type": "watcher_done", "why": f"the game refused it for {patience} s"})
            else:
                started = None                    # closed again before its audio started: wait for the next start
            time.sleep(poll)
    except Exception as e:  # noqa: BLE001
        log_route({"type": "watcher_error", "error": repr(e)})


def status():
    st, procs = load(), processes()
    pids = game_pids(st.get("process", GAME), procs)
    dev = ""
    if pids:
        try:
            dev = Policy().get(pids[0])
        except OSError as e:
            dev = f"? ({e})"
    names = {}
    try:                                           # device names, when the tools env's pycaw is at hand
        import live_devices
        names = {o.id: o.name for o in live_devices.outputs()}
    except Exception:  # noqa: BLE001
        pass
    return {"game": st.get("process", GAME), "game_running": bool(pids),
            "game_device": names.get(dev, dev) if dev else (DEFAULT if pids else ""),
            "active": bool(st.get("active")), "pending": bool(st.get("pending")),
            "to": st.get("to_name", ""), "back": st.get("back_name", ""), "since": st.get("since", ""),
            "watcher": _alive(st.get("watcher_pid"), procs), "startup_item": LINK.exists(),
            "live": live_running(procs), "dry": DRY}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "watch":
        watch()
    elif cmd in ("restore", "reset"):
        r, back = restore("manual") if cmd == "restore" else reset()
        print(json.dumps({"result": r, "back": back}, ensure_ascii=False))
    elif cmd == "status":
        print(json.dumps(status(), ensure_ascii=False))
    else:
        raise SystemExit(__doc__)
