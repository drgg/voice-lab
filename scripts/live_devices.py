"""Device discovery for the live tool (Windows, tools env): which output the game plays to, which output you
listen on, and what changes while running (headphones plugged or unplugged, the Windows default changed, the
game's output moved in the volume mixer).

  capture  the output device holding the game's audio session (asked from Windows, by process name), else a
           virtual device carrying sound, else the first virtual device (the game is not running yet)
  play     the Windows default output if it is a real device you can hear; otherwise the best real device:
           headphones, then speakers, then a display (HDMI), then a digital output. Never a virtual device,
           never a Bluetooth hands-free (telephony) endpoint, never the device being captured
Run it alone to see what the live tool would pick:  envs/tools/Scripts/python scripts/live_devices.py
"""
import os
import sys
from dataclasses import dataclass, field

# soundcard puts its threads in COM's multithreaded apartment; comtypes would otherwise ask for a single-threaded
# one on import and fail ("cannot change thread mode"). pycaw works in either.
sys.coinit_flags = getattr(sys, "coinit_flags", 0)

GAME = os.environ.get("VL_GAME_PROCESS", "")  # the app whose sound is captured
VIRTUAL = ("steam streaming", "cable input", "cable output", "cable-a", "cable-b", "voicemeeter", "vb-audio",
           "virtual audio", "nvidia broadcast")
HANDSFREE = ("hands-free", "handsfree", "免提", "headset audio gateway")
RANK = [("headphones", ("耳机", "headphone", "headset", "earphone", "airpods", "buds")),
        ("speakers", ("扬声器", "speaker", "realtek(r) audio")),
        ("display", ("monitor", "hdmi", "displayport", "nvidia high definition", "amd high definition")),
        ("digital", ("digital", "spdif", "s/pdif", "optical"))]


@dataclass
class Out:
    id: str
    name: str
    default: bool = False
    sessions: list = field(default_factory=list)      # (process name, "active" | "inactive")

    @property
    def virtual(self):
        n = self.name.lower()
        return any(v in n for v in VIRTUAL)

    @property
    def handsfree(self):
        n = self.name.lower()
        return any(v in n for v in HANDSFREE)

    @property
    def kind(self):
        n = self.name.lower()
        for k, keys in RANK:
            if any(w in n for w in keys):
                return k
        return "other"

    def has(self, process):
        return [s for s in self.sessions if s[0].lower() == process.lower()]


def outputs():
    """Active render endpoints with their audio sessions. COM: call from one thread (it initialises COM)."""
    import comtypes
    import psutil
    from comtypes import CLSCTX_ALL
    from pycaw.api.mmdeviceapi import IMMDeviceEnumerator
    from pycaw.constants import CLSID_MMDeviceEnumerator, DEVICE_STATE, EDataFlow
    from pycaw.pycaw import AudioUtilities, IAudioSessionControl2, IAudioSessionManager2

    try:
        comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
    except OSError:
        pass                                            # already initialised in this thread
    enum = comtypes.CoCreateInstance(CLSID_MMDeviceEnumerator, IMMDeviceEnumerator, comtypes.CLSCTX_INPROC_SERVER)
    coll = enum.EnumAudioEndpoints(EDataFlow.eRender.value, DEVICE_STATE.ACTIVE.value)
    try:
        default = enum.GetDefaultAudioEndpoint(EDataFlow.eRender.value, 1).GetId()    # eMultimedia
    except Exception:
        default = None
    outs = []
    for i in range(coll.GetCount()):
        dev = coll.Item(i)
        o = Out(dev.GetId(), AudioUtilities.CreateDevice(dev).FriendlyName)
        o.default = o.id == default
        try:
            mgr = dev.Activate(IAudioSessionManager2._iid_, CLSCTX_ALL, None).QueryInterface(IAudioSessionManager2)
            se = mgr.GetSessionEnumerator()
            for j in range(se.GetCount()):
                ctl = se.GetSession(j).QueryInterface(IAudioSessionControl2)
                pid, state = ctl.GetProcessId(), ctl.GetState()
                if state == 2 or not pid:               # expired, or the system sounds session
                    continue
                try:
                    pn = psutil.Process(pid).name()
                except Exception:
                    continue
                o.sessions.append((pn, "active" if state == 1 else "inactive"))
        except Exception:
            pass
        outs.append(o)
    return outs


def endpoint_volume(dev_id):
    """(dB, muted) of an output device's own volume slider. A loopback capture is taken after it: a device
    at 26 % hands the game over 20 dB down, so the live tool adds this back."""
    import comtypes
    from comtypes import CLSCTX_ALL
    from pycaw.api.endpointvolume import IAudioEndpointVolume
    from pycaw.api.mmdeviceapi import IMMDeviceEnumerator
    from pycaw.constants import CLSID_MMDeviceEnumerator

    try:
        comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
    except OSError:
        pass
    enum = comtypes.CoCreateInstance(CLSID_MMDeviceEnumerator, IMMDeviceEnumerator, comtypes.CLSCTX_INPROC_SERVER)
    ev = enum.GetDevice(dev_id).Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None).QueryInterface(
        IAudioEndpointVolume)
    return float(ev.GetMasterVolumeLevel()), bool(ev.GetMute())


def match(outs, name):
    m = [o for o in outs if name.lower() in o.name.lower()]
    return m[0] if m else None


def choose_capture(outs, process=GAME, override="", levels=None):
    """(Out, reason). levels: {id: dBFS} of the virtual devices' loopback, if measured."""
    if override and override != "auto":
        o = match(outs, override)
        return (o, "named") if o else (None, f"no output device matches '{override}'")
    # where Windows has the game's audio sessions. The game can hold several: its sound follows the per-app
    # setting (a virtual device) while another, silent stream stays on the default output; so a virtual device
    # with a session wins, and only a game whose sessions are all on real devices counts as misrouted.
    for state in ("active", "inactive"):
        hits = [o for o in outs if any(s[1] == state for s in o.has(process))]
        hits.sort(key=lambda o: 0 if o.virtual and not o.handsfree else 1)
        if hits:
            return hits[0], f"{process} plays here"
    virt = [o for o in outs if o.virtual and not o.handsfree]
    if levels:
        loud = sorted((levels.get(o.id, -200), i, o) for i, o in enumerate(virt))
        if loud and loud[-1][0] > -90:
            return loud[-1][2], "virtual device carrying sound"
    pref = sorted(virt, key=lambda o: (0 if "steam streaming speakers" in o.name.lower() else
                                       1 if "cable input" in o.name.lower() else 2))
    if pref:
        return pref[0], f"{process} not running: waiting on the usual virtual device"
    return None, "no virtual output device (Steam Streaming Speakers or VB-CABLE) found"


def choose_play(outs, capture, override=""):
    """(Out, reason): where you listen."""
    cid = capture.id if capture else None
    if override and override != "auto":
        o = match([x for x in outs if x.id != cid], override)
        return (o, "named") if o else (None, f"no other output device matches '{override}'")
    real = [o for o in outs if not o.virtual and not o.handsfree and o.id != cid]
    for o in real:
        if o.default:
            return o, "Windows default output"
    order = [k for k, _ in RANK] + ["other"]
    real.sort(key=lambda o: order.index(o.kind))
    if real:
        return real[0], f"best connected device ({real[0].kind})"
    return None, "no real output device connected"


def misrouted(capture):
    """The game plays straight to a device you hear: replacing would only add a second voice."""
    return capture is not None and not capture.virtual


def describe(outs, cap, play):
    rows = []
    for o in outs:
        tags = []
        if o.default:
            tags.append("Windows default")
        if o.virtual:
            tags.append("virtual")
        if o.handsfree:
            tags.append("hands-free (telephony)")
        if not o.virtual and not o.handsfree:
            tags.append(o.kind)
        if o.has(GAME):
            tags.append("game plays here")
        role = " <- capture" if cap and o.id == cap.id else " <- play" if play and o.id == play.id else ""
        rows.append(f"  {o.name}  [{', '.join(tags)}]{role}")
    return "\n".join(rows)


if __name__ == "__main__":
    outs = outputs()
    cap, why_c = choose_capture(outs)
    play, why_p = choose_play(outs, cap)
    print(describe(outs, cap, play))
    print(f"capture: {cap.name if cap else '-'} ({why_c})")
    print(f"play:    {play.name if play else '-'} ({why_p})")
    if misrouted(cap):
        print(f"! {GAME} plays straight to {cap.name}: right without the live tool; the live tool switches it to "
              "the virtual device by itself (app_route.py), or with --route off set it in Windows' volume mixer")
