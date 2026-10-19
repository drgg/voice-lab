"""Clean the target clips used as references: de-reverb, then de-noise (tools env).

Zero-shot models copy the reference's acoustic environment, so leftover BGM / room reverb / hiss in
the anime clips turns into noise in the converted lines. Every clip that can enter the reference
bank is passed through a MelBand Roformer de-reverb model and the UVR de-noise model; results go to
data/<voice>/slices_clean/<id>.wav (same ids). Already-cleaned clips are skipped.

Usage:  python scripts/clean_refs.py [--min-score 0.55] [--max-bg -8]
        python scripts/clean_refs.py --ids-csv data/<voice>/diar/segclips.csv   (every id in that csv instead)
"""
import argparse
import shutil
from pathlib import Path

from vl_common import CACHE, TARGET, is_target_clip, read_csv

CLEAN = TARGET / "slices_clean"
WORK = CACHE / "tmp" / "clean_refs"
STEPS = [  # (model file, substring of the stem to keep)
    ("dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt", "(noreverb)"),
    ("UVR-DeNoise.pth", "(no noise)"),
]


def run_step(sep, src, keep, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    produced = [out_dir / Path(p).name for p in sep.separate(str(src))]
    kept = next(p for p in produced if keep in p.name.lower())
    for p in produced:
        if p != kept:
            p.unlink(missing_ok=True)
    return kept


if __name__ == "__main__":
    from audio_separator.separator import Separator

    ap = argparse.ArgumentParser()
    ap.add_argument("--min-p", type=float, default=0.7)
    ap.add_argument("--min-score", type=float, default=0.6)
    ap.add_argument("--ids-csv", default="", help="clean the clips listed there (an id column; target_segments.py "
                                                  "clips) instead of the ranked clips that pass is_target_clip")
    a = ap.parse_args()
    if a.ids_csv:
        rows = [r for r in read_csv(Path(a.ids_csv)) if not (CLEAN / f"{r['id']}.wav").exists()]
    else:
        rows = [
            r for r in read_csv(TARGET / "slices_ranked.csv")
            if is_target_clip(r, a.min_p, a.min_score) and not (CLEAN / f"{r['id']}.wav").exists()
        ]
    print(f"{len(rows)} clips to clean")
    CLEAN.mkdir(parents=True, exist_ok=True)
    current = [TARGET / "slices" / f"{r['id']}.wav" for r in rows]
    for k, (model, keep) in enumerate(STEPS):
        out_dir = WORK / f"step{k}"
        sep = Separator(output_dir=str(out_dir), model_file_dir=str(CACHE / "separator"), output_format="WAV")
        sep.load_model(model_filename=model)
        current = [run_step(sep, p, keep, out_dir) for p in current]
        print(f"step {k + 1}/{len(STEPS)} {model}: done")
    for r, p in zip(rows, current):
        shutil.move(str(p), CLEAN / f"{r['id']}.wav")
    shutil.rmtree(WORK, ignore_errors=True)
    print(f"cleaned {len(rows)} clips -> {CLEAN}")
