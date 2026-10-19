"""Where the current line set and target voice live. Standard library only: imported by the scripts of every env.

VL_SET (environment variable; run_d4.ps1 -Set sets it) picks the line set:
  unset / empty   the 20 experiment lines   data/source/lines   exp/         eval/
  <name>          a set from prep_set.py    data/source/sets/<name>/lines   exp/<name>/   eval/<name>/
Beside lines/ every set has phrases/ (segment_lines.py), refine/ (refine_split.py) and routes.csv
(scan_lines.py); finished files are exported to out/<name>/ (finalize.py).

VL_VOICE (run_d4.ps1 -Voice sets it) picks the target voice: its material lives in data/<voice>/ (episodes'
audio, clips, training set, references, voice.json). The first voice, default, keeps the original folders;
any other voice gets its own results beside them: exp/<set>_<voice>/, eval/<set>_<voice>/, out/<set>_<voice>/.
The line set's own analysis (lines, phrases, routes) and the game-side results (recordings, voice path) are
shared by every voice: EXP_BASE.

VL_CHAR (run_d4.ps1 -Char sets it) picks the game character whose lines are replaced: data/<char>/ holds its
exported voice files (wem/), decoded lines and sets. The first one, source, keeps the folders above; another one puts
its character name in front of the set's: data/<char>/sets/<set>/, exp/<char>_<set>[_<voice>]/, eval/..., out/...
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SET = os.environ.get("VL_SET", "").strip()
SET_NAME = SET or "test20"
FIRST_VOICE = "default"                  # keeps the original folders (exp/<set>, eval/<set>)
VOICE = os.environ.get("VL_VOICE", "").strip() or FIRST_VOICE
VOICE_DIR = ROOT / "data" / VOICE
FIRST_CHAR = "source"                    # keeps the original folders too
CHAR = os.environ.get("VL_CHAR", "").strip() or FIRST_CHAR
CHAR_DIR = ROOT / "data" / CHAR
SET_DIR = CHAR_DIR / "sets" / SET if SET else CHAR_DIR
LINES = SET_DIR / "lines"
PHRASES = SET_DIR / "phrases"
REFINE = SET_DIR / "refine"
ROUTES = SET_DIR / "routes.csv"
_FIRST = CHAR == FIRST_CHAR
_BASE = SET_NAME if _FIRST else f"{CHAR}_{SET_NAME}"  # this character's set, as folder names carry it
EXP_BASE = (ROOT / "exp" / SET if SET else ROOT / "exp") if _FIRST else ROOT / "exp" / _BASE
_TAG = "" if VOICE == FIRST_VOICE else f"_{VOICE}"
EXP = EXP_BASE if not _TAG else ROOT / "exp" / (_BASE + _TAG)
EVAL = ((ROOT / "eval" / SET if SET else ROOT / "eval") if _FIRST else ROOT / "eval" / _BASE) if not _TAG \
    else ROOT / "eval" / (_BASE + _TAG)
REFS = VOICE_DIR / "refs" / "sets" / (SET if _FIRST else _BASE) if SET else VOICE_DIR / "refs"
OUT = ROOT / "out" / (_BASE + _TAG)
# the game's voice chain (live tool, recording replacement): measured on a recording of the first character's
# lines; the same mixer carries every character, so another character uses it until it has its own
VOICE_PATH_FIR = EXP_BASE / "voice_path_fir.npy"
if not VOICE_PATH_FIR.exists():
    VOICE_PATH_FIR = (ROOT / "exp" / SET if SET else ROOT / "exp") / "voice_path_fir.npy"

def char_paths(char, set_name="all", voice=FIRST_VOICE):
    """The folders of another character's set and voice, computed as the constants above are for the current
    VL_CHAR / VL_SET / VL_VOICE - for one process serving several characters (the live tool, 2026-10-16)."""
    first = char == FIRST_CHAR
    name = set_name or "test20"
    base = name if first else f"{char}_{name}"
    exp_base = (ROOT / "exp" / set_name if set_name else ROOT / "exp") if first else ROOT / "exp" / base
    exp = exp_base if voice == FIRST_VOICE else ROOT / "exp" / f"{base}_{voice}"
    fir = exp_base / "voice_path_fir.npy"
    if not fir.exists():  # the same mixer carries every character (see VOICE_PATH_FIR)
        fir = (ROOT / "exp" / set_name if set_name else ROOT / "exp") / "voice_path_fir.npy"
    set_dir = ROOT / "data" / char / "sets" / set_name if set_name else ROOT / "data" / char
    return {"lines": set_dir / "lines", "exp": exp, "exp_base": exp_base, "voice_path": fir}


# per-voice settings (data/<voice>/voice.json); the defaults are the first voice's
_VOICE_DEFAULTS = {
    "label": "target",                       # name on pages
    "own_markers": "",          # regex: the character's own speech habits (weak positive labels)
    "other_markers": r"私|わたし|あたし|わよ|です|ます|でした|ました|でしょうか|ござい|俺",  # other characters'
    "keyword": "",                # catchphrase only she says: anchors for the first ranking
    "min_f0": 180,                       # Hz: clips below this median pitch are not her (male characters)
    "nonverbal_pitch": 2,                # semitones from the game character's register to hers
}


def voice_cfg():
    p = VOICE_DIR / "voice.json"
    cfg = dict(_VOICE_DEFAULTS)
    if p.exists():
        cfg.update(json.loads(p.read_text(encoding="utf-8")))
    return cfg


def char_label():
    """The game character's name on pages: data/<char>/char.json {"label": ...}; the first one is 原角色."""
    p = CHAR_DIR / "char.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8")).get("label", CHAR)
    return "原角色" if _FIRST else CHAR


def page_path(path, page_dir=None):
    """URL of `path` relative to a page in `page_dir` (default EVAL), for the local page server."""
    return os.path.relpath(path, page_dir or EVAL).replace(os.sep, "/")


def flagged_clips():
    """Ids of the target voice's clips that clip_purity.py (another voice inside) or transcript_check.py (two ASRs
    disagree) flagged: rows with a flag in data/<voice>/purity.csv and transcripts_check.csv (empty before they ran).
    the voice's 25 were found after its first model was trained (2026-10-15); a new voice leaves them out from the start."""
    import csv

    out = set()
    for name in ("purity.csv", "transcripts_check.csv"):
        p = VOICE_DIR / name
        if p.exists():
            with open(p, newline="", encoding="utf-8-sig") as f:
                out |= {r["id"] for r in csv.DictReader(f) if r.get("flag")}
    # clips the user heard as someone else (speaker_labels.csv, label "other"): whole clips of another woman pass
    # both checks above - 26 of the target voice's (2026-10-18: Miku, the Klaxosaur Princess, ...)
    p = VOICE_DIR / "speaker_labels.csv"
    if p.exists():
        with open(p, newline="", encoding="utf-8-sig") as f:
            out |= {r["id"] for r in csv.DictReader(f) if r.get("label") == "other"}
    return out


def routes():
    """id -> route ("speech" / "refine" / "nonverbal") from scan_lines.py; {} before the first scan."""
    if not ROUTES.exists():
        return {}
    import csv

    with open(ROUTES, newline="", encoding="utf-8-sig") as f:
        return {r["id"]: r["route"] for r in csv.DictReader(f)}
