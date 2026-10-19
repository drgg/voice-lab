"""Who is in the party, read off the game's party panel on screen (tools env) - for the live tool, which then
matches only those characters' lines (the user, 2026-10-17: 根据角色是否在当前队伍进行语音的取舍; 选用方案2).

The panel (right edge, three slots, 264 px apart at 3840x2160): a slot in the background shows a half-length
portrait with its number badge (1/2/3) at the left; the character on the field shows a slightly paler portrait in
a ring and no badge. Both look alike in the face area, which is compared with the character's face templates
(colour NCC on 40x40). Measured on the user's four 4K screenshots: the same character 0.82-0.99 (field or
background, any slot), other characters at most 0.59.

  the HUD is shown   at least one number badge matches its template (NCC >= BADGE_MIN); in cutscenes, menus and
                     the map (no badge) every character is switched in (Follow)
  a slot             a known character at >= MATCH; "other" below OTHER (someone else, or empty); else unsure
  the party          known characters recognised in a slot; anyone else is out only when no slot is unsure

Only the screen is read (a GDI copy of the 0.2-megapixel strip, every 2 s; see _screen_rect): nothing in the game
is touched. Geometry is stored for 3840x2160 and scaled with the screen height (16:9, UI anchored right).

  python scripts/party_screen.py build --shots <dir> --label 1:B,2:A*,3:o ...   (see build())
  python scripts/party_screen.py read --image <png>    or    read    (the screen now)
"""
import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
from PIL import Image

from vl_common import ROOT

UI = ROOT / "data" / "party_ui"
REF_W, REF_H = 3840, 2160
BAR_Y = [575, 839, 1103]                 # health bar under each slot (4K), 264 px apart
FACE = (3522, -150, 3640, -30)           # x0, dy0, x1, dy1: the face, clear of badge and element icons
BADGE = (3469, -153, 3512, -110)
MATCH, OTHER, BADGE_MIN = 0.75, 0.60, 0.80
STRIP = (3440, 400, 3700, 1115)          # what is captured (4K): all three slots


def _vec(tile, size=40):
    a = np.asarray(tile.convert("RGB").resize((size, size), Image.BILINEAR)).astype(np.float32).reshape(-1, 3)
    a -= a.mean(0)
    a /= np.linalg.norm(a, axis=0) + 1e-6
    return a.ravel() / np.sqrt(3)


def _gray(tile, size=24):
    a = np.asarray(tile.convert("L").resize((size, size), Image.BILINEAR)).astype(np.float32).ravel()
    a -= a.mean()
    return a / (np.linalg.norm(a) + 1e-6)


def _boxes(img_w, img_h, origin=(0, 0)):
    """Face and badge boxes of the three slots in an image of the whole screen (or a strip from `origin`)."""
    s = img_h / REF_H if not origin[1] else 1.0
    out = []
    for y in BAR_Y:
        f = (FACE[0], y + FACE[1], FACE[2], y + FACE[3])
        b = (BADGE[0], y + BADGE[1], BADGE[2], y + BADGE[3])
        out.append(tuple(tuple(int(round(v * s)) - (origin[0] if i % 2 == 0 else origin[1]) for i, v in enumerate(box))
                         for box in (f, b)))
    return out


class Reader:
    def __init__(self, ui=UI):
        cfg = json.loads((ui / "templates.json").read_text(encoding="utf-8"))
        self.faces = {}                                   # char -> [vectors]
        for t in cfg["faces"]:
            self.faces.setdefault(t["char"], []).append(_vec(Image.open(ui / t["file"])))
        self.badges = [_gray(Image.open(ui / f)) for f in cfg["badges"]]
        self.chars = list(self.faces)

    def read(self, img, origin=(0, 0)):
        """img: the screen (or the strip captured from `origin`, at 4K scale). -> {"hud": bool, "slots": [...]}"""
        slots, badges = [], 0
        for face, badge in _boxes(*img.size, origin=origin):
            bs = max(float(_gray(img.crop(badge)) @ t) for t in self.badges)
            badges += bs >= BADGE_MIN
            v = _vec(img.crop(face))
            sc = {c: max(float(v @ t) for t in ts) for c, ts in self.faces.items()}
            best = max(sc, key=sc.get)
            who = best if sc[best] >= MATCH else "other" if sc[best] < OTHER else "?"
            slots.append({"who": who, "score": round(sc[best], 3), "badge": round(bs, 2)})
        return {"hud": badges >= 1, "slots": slots}

    def party(self, r):
        """(in, out): characters recognised in a slot, and the known ones certainly not there (None: unknown)."""
        if not r["hud"]:
            return None
        here = {s["who"] for s in r["slots"]} & set(self.chars)
        if any(s["who"] == "?" for s in r["slots"]):
            return here, set()
        return here, set(self.chars) - here


class Follow:
    """Readings -> party changes for the live tool: a reading counts once it repeats (one frame of an animation or
    a menu fading in does not switch anyone), and acts only when it differs from the last one acted on, so a
    change made by hand (Ctrl+Alt+<n>) stays until the party on screen changes."""

    def __init__(self, chars):
        self.known, self.prev, self.acted = set(chars), None, None

    def step(self, p):
        """p: Reader.party(...) of one reading. Returns (in, out) to apply, or None. With the panel hidden (menus,
        cutscenes, the map) every character is in: a character's profile page plays her lines whether she is in the
        party or not, and a character switched off lets the others' lines be taken for hers - replaying the
        the second character stream with her off, 4 source lines were put over the second character's voice at ncc 0.71-0.85."""
        now = (frozenset(self.known), frozenset()) if p is None else (frozenset(p[0] & self.known),
                                                                        frozenset(p[1] & self.known))
        act = now if now == self.prev and now != self.acted else None
        if act is not None:
            self.acted = act
        self.prev = now
        return act


def screen_size():
    u = ctypes.windll.user32
    return u.GetSystemMetrics(0), u.GetSystemMetrics(1)


def dpi_aware():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)    # physical pixels, whatever the display scaling
    except (OSError, AttributeError):
        ctypes.windll.user32.SetProcessDPIAware()


class _BMI(ctypes.Structure):
    _fields_ = [("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32), ("biHeight", ctypes.c_int32),
                ("biPlanes", ctypes.c_uint16), ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
                ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
                ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32), ("biClrImportant", ctypes.c_uint32),
                ("bmiColors", ctypes.c_uint32 * 3)]


def _gdi():
    u, g = ctypes.windll.user32, ctypes.windll.gdi32
    H = ctypes.c_void_p
    for f, res, args in ((u.GetDC, H, [H]), (u.ReleaseDC, ctypes.c_int, [H, H]), (g.CreateCompatibleDC, H, [H]),
                         (g.CreateCompatibleBitmap, H, [H, ctypes.c_int, ctypes.c_int]), (g.SelectObject, H, [H, H]),
                         (g.BitBlt, ctypes.c_int, [H, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, H,
                                                   ctypes.c_int, ctypes.c_int, ctypes.c_uint32]),
                         (g.GetDIBits, ctypes.c_int, [H, H, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
                                                      ctypes.c_void_p, ctypes.c_uint]),
                         (g.DeleteObject, ctypes.c_int, [H]), (g.DeleteDC, ctypes.c_int, [H])):
        f.restype, f.argtypes = res, args
    return u, g


def _screen_rect(x, y, w, h):
    """RGB image of a rectangle of the screen, copied by GDI. Only the rectangle is copied, and each ctypes call
    lets other Python threads run: PIL's ImageGrab copies the whole screen while holding the GIL, and in the live
    tool that stopped the capture thread for 56-115 ms every 2 s (2026-10-18: the WASAPI buffer overflowed, the
    sound gained 30-75 ms holes and the lines being replaced were lost)."""
    u, g = _gdi()
    sdc = u.GetDC(None)
    mdc = g.CreateCompatibleDC(sdc)
    bmp = g.CreateCompatibleBitmap(sdc, w, h)
    old = g.SelectObject(mdc, bmp)
    try:
        g.BitBlt(mdc, 0, 0, w, h, sdc, x, y, 0x00CC0020)             # SRCCOPY
        bi = _BMI(biSize=40, biWidth=w, biHeight=-h, biPlanes=1, biBitCount=32, biCompression=0)  # top-down
        buf = ctypes.create_string_buffer(w * h * 4)
        if g.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(bi), 0) != h:
            raise OSError("GetDIBits failed")
    finally:
        g.SelectObject(mdc, old)
        g.DeleteObject(bmp)
        g.DeleteDC(mdc)
        u.ReleaseDC(None, sdc)
    return Image.frombuffer("RGB", (w, h), buf.raw, "raw", "BGRX", 0, 1)


def grab():
    """The panel strip of the primary screen, scaled to 4K coordinates: (image, origin)."""
    w, h = screen_size()
    s = h / REF_H
    x0, y0, x1, y1 = STRIP
    box = (int(w - (REF_W - x0) * s), int(y0 * s), int(w - (REF_W - x1) * s), int(y1 * s))  # anchored right
    img = _screen_rect(box[0], box[1], box[2] - box[0], box[3] - box[1])
    if s != 1.0:
        img = img.resize((x1 - x0, y1 - y0), Image.BILINEAR)
    return img, (x0, y0)


def build(a):
    """Templates from the user's screenshots: --label "<shot>:<slot>:<char>" for every slot to learn from, e.g.
    1:1:other (shot 1 = the first .png in --shots by name). Badges: --badge "<shot>:<slot>"."""
    shots = sorted(Path(a.shots).glob("*.png"))
    UI.mkdir(parents=True, exist_ok=True)
    cfg = {"source": str(Path(a.shots)), "screen": [REF_W, REF_H], "faces": [], "badges": []}
    for spec in a.label:
        n, slot, char = spec.split(":")
        img = Image.open(shots[int(n) - 1]).convert("RGB")
        face, _ = _boxes(*img.size)[int(slot) - 1]
        f = f"{char}_{n}_{slot}.png"
        img.crop(face).save(UI / f)
        cfg["faces"].append({"char": char, "file": f, "shot": shots[int(n) - 1].name, "slot": int(slot)})
    for spec in a.badge:
        n, slot = spec.split(":")
        img = Image.open(shots[int(n) - 1]).convert("RGB")
        _, b = _boxes(*img.size)[int(slot) - 1]
        f = f"badge_{slot}.png"
        img.crop(b).save(UI / f)
        cfg["badges"].append(f)
    (UI / "templates.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(cfg['faces'])} faces, {len(cfg['badges'])} badges -> {UI}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build")
    s.add_argument("--shots", required=True)
    s.add_argument("--label", action="append", default=[])
    s.add_argument("--badge", action="append", default=[])
    s.set_defaults(fn=build)
    s = sub.add_parser("read")
    s.add_argument("--image", default="")
    a = ap.parse_args()
    if a.cmd == "build":
        a.fn(a)
    else:
        rd = Reader()
        if a.image:
            r = rd.read(Image.open(a.image).convert("RGB"))
        else:
            dpi_aware()
            img, origin = grab()
            r = rd.read(img, origin)
        print(json.dumps(r, ensure_ascii=False), "->", rd.party(r))
