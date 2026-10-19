"""Held-out test set for a text-to-speech model of the target voice (VL_VOICE; tools env).

The model is judged by re-dubbing her own clips from their text and comparing with the real ones (tts_eval.py),
so those clips stay out of its training. Picks --n clips from the voice's training set (data/<voice>/train.csv),
1.5-7 s, one sentence (no pause gap in the transcript), confidently hers (p_target >= 0.8), at most --per-ep per
episode, spread round-robin over emotion2vec's top category x arousal tercile x question-or-not. Training clips of
the same episode within --guard s of a test clip are held out too (same scene, often the same words).

Each test clip gets style references from the rest: emotion_match.py's precise rule, its own audio standing in
for what the game line is later (it only chooses which of her other clips to imitate). References are 3.2-9.5 s
(GPT-SoVITS takes 3-10 s prompts) and from another scene (not the same episode within --ref-guard s).
Needs the bank caches: emotion_match.py --bank-only.

Writes data/<voice>/tts/testset.csv (key T01..., test clip, reference + 2 alternates), holdout_ids.txt and
speak_test.csv (key, text, ref_path, ref_text: all a system may use).
  set VL_VOICE=myvoice & python scripts/tts_testset.py [--n 60]
"""
import argparse
import csv
import random
import re
from pathlib import Path

import numpy as np

from emotion_match import AROUSAL, clip_path, precise_scores, sentence_type
from vl_common import TARGET, read_csv, write_csv

def episode_key(cid):
    """'S01E05' style or the bare episode number - only used to group clips of one episode."""
    return cid.rsplit("_", 1)[0]


def start_s(cid):
    return int(re.sub(r"[a-z]+$", "", cid.rsplit("_", 1)[1])) / 1000  # x / d: cuts of another release / DiariZen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--per-ep", type=int, default=3)
    ap.add_argument("--guard", type=float, default=10.0, help="s around a test clip kept out of training")
    ap.add_argument("--ref-guard", type=float, default=60.0, help="s around a test clip no reference comes from")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--exclude-flagged", action="store_true",
                    help="no test clip or reference among the clips clip_purity.py / transcript_check.py flagged")
    ap.add_argument("--keep", default="",
                    help="an earlier testset.csv: keep its test clips and keys (minus --drop) and only pick their "
                         "references again, from the current bank (after the training set was rebuilt)")
    ap.add_argument("--drop", default="", help="--keep: keys to leave out, e.g. T14,T22 (clips found not to be hers)")
    ap.add_argument("--test-arousal", default="",
                    help="--keep: an arousal_bank csv that still has the kept clips' cues (the bank before the rebuild)")
    a = ap.parse_args()
    out = TARGET / "tts"
    out.mkdir(parents=True, exist_ok=True)

    train = read_csv(TARGET / "train.csv")
    z = np.load(TARGET / "emo_bank.npz")
    emo = {str(c): (z["feats"][i], str(z["labels"][i]), z["scores"][i]) for i, c in enumerate(z["ids"])}
    num = lambda v: float(v) if v not in ("", None) else np.nan  # noqa: E731
    ar_rows = read_csv(TARGET / "arousal_bank.csv")
    A = np.array([[num(r[k]) for k in AROUSAL] for r in ar_rows])
    A = (A - np.nanmean(A, 0)) / (np.nanstd(A, 0) + 1e-9)
    aro = {r["id"]: v for r, v in zip(ar_rows, A)}
    bad = set()
    if a.exclude_flagged:
        from vl_paths import flagged_clips

        bad = flagged_clips()
    bank = [r for r in train if r["id"] in emo and r["id"] in aro and r["id"] not in bad]
    print(f"{len(train)} training clips, {len(bank)} with emotion + arousal cues" + (f" ({len(bad)} flagged left out)" if bad else ""))

    if a.keep:
        drop = {k.strip() for k in a.drop.split(",") if k.strip()}
        test = [{"id": r["id"], "duration": r["duration"], "text": r["text"], "key": r["key"]}
                for r in read_csv(Path(a.keep)) if r["key"] not in drop]
        # their cues: emotion from the cache (it keeps every clip it ever embedded), arousal from the old bank file,
        # z-scored with the current bank's spread like the bank itself
        raw = {r["id"]: np.array([num(r[k]) for k in AROUSAL]) for r in read_csv(Path(a.test_arousal))}
        A0 = np.array([[num(r[k]) for k in AROUSAL] for r in ar_rows])
        mu, sd = np.nanmean(A0, 0), np.nanstd(A0, 0) + 1e-9
        for r in test:
            aro.setdefault(r["id"], (raw[r["id"]] - mu) / sd)
        missing = [r["id"] for r in test if r["id"] not in emo]
        if missing:
            raise SystemExit(f"{len(missing)} kept test clips have no emotion embedding, e.g. {missing[0]}")
        level = {r["id"]: np.nanmean(aro[r["id"]][[0, 1, 2, 4]]) for r in test}
        t1, t2 = np.percentile([np.nanmean(aro[r["id"]][[0, 1, 2, 4]]) for r in bank], [33.3, 66.7])
        per_ep = {}
        for r in test:
            per_ep[episode_key(r["id"])] = per_ep.get(episode_key(r["id"]), 0) + 1
        print(f"kept {len(test)} test clips of {a.keep} (left out: {', '.join(sorted(drop)) or 'none'})")
    else:
        def ok_test(r):
            t = r["text"].strip()
            return (1.5 <= float(r["duration"]) <= 7 and float(r["p_target"]) >= 0.8 and " " not in t and "　" not in t
                    and len(re.sub(r"[^぀-ヿ一-鿿]", "", t)) >= 5)

        cands = [r for r in bank if ok_test(r)]
        level = {r["id"]: np.nanmean(aro[r["id"]][[0, 1, 2, 4]]) for r in cands}  # pitch, range, loudness, effort
        t1, t2 = np.percentile(list(level.values()), [33.3, 66.7])
        cat = lambda r: emo[r["id"]][1] if emo[r["id"]][1] in ("angry", "happy", "surprised", "fearful") else "other"  # noqa: E731
        strata = {}
        for r in cands:
            k = (cat(r), 0 if level[r["id"]] < t1 else 1 if level[r["id"]] < t2 else 2,
                 sentence_type(r["text"]) == "question")
            strata.setdefault(k, []).append(r)
        rng = random.Random(a.seed)
        for v in strata.values():
            rng.shuffle(v)
        keys = sorted(strata, key=lambda k: -len(strata[k]))
        test, per_ep = [], {}
        while len(test) < a.n and any(strata[k] for k in keys):
            for k in keys:
                while strata[k]:
                    r = strata[k].pop()
                    if per_ep.get(episode_key(r["id"]), 0) < a.per_ep:
                        test.append(r)
                        per_ep[episode_key(r["id"])] = per_ep.get(episode_key(r["id"]), 0) + 1
                        break
                if len(test) >= a.n:
                    break
    test_ids = {r["id"] for r in test}

    def near(r, guard):
        return [x["id"] for x in train if episode_key(x["id"]) == episode_key(r["id"])
                and abs(start_s(x["id"]) - start_s(r["id"])) < guard]

    holdout = set(test_ids)
    for r in test:
        holdout.update(near(r, a.guard))
    # references: the precise rule over the clips still in training, 3-10 s, not from the test clip's scene
    pool = [r for r in bank if r["id"] not in holdout and 3.2 <= float(r["duration"]) <= 9.5]  # margin
    B_e = np.stack([emo[r["id"]][0] for r in pool])
    B_s = np.stack([emo[r["id"]][2] for r in pool])
    B_a = np.stack([aro[r["id"]] for r in pool])
    B_t = np.array([sentence_type(r["text"]) for r in pool])
    mean_bank = np.mean([v[0] for v in emo.values()], 0)
    rows = []
    for r in test:
        total, parts = precise_scores(emo[r["id"]][0], emo[r["id"]][2], aro[r["id"]], sentence_type(r["text"]),
                                      B_e, B_s, B_a, B_t, mean_bank, mean_bank)
        scene = set(near(r, a.ref_guard))
        order = [j for j in np.argsort(-total) if pool[j]["id"] not in scene][:3]
        best = pool[order[0]]
        rows.append({"key": r.get("key") or f"T{len(rows) + 1:02d}", "id": r["id"], "file": clip_path(r["id"]).name, "duration": r["duration"], "text": r["text"],
                     "emotion": emo[r["id"]][1], "arousal": round(float(level[r["id"]]), 2),
                     "type": sentence_type(r["text"]), "ref_id": best["id"], "ref_text": best["text"],
                     "ref_duration": best["duration"], "ref_emotion": emo[best["id"]][1],
                     "ref_score": round(float(total[order[0]]), 2), "ref_arousal_dist": round(float(parts["aro"][order[0]]), 2),
                     "alt_refs": " | ".join(pool[j]["id"] for j in order[1:])})
    write_csv(out / "testset.csv", rows)
    # what a text-to-speech system gets: the words and the style reference, never the test clip itself
    write_csv(out / "speak_test.csv", [{"key": r["key"], "text": r["text"], "ref_path": str(clip_path(r["ref_id"])),
                                        "ref_text": r["ref_text"]} for r in rows])
    (out / "holdout_ids.txt").write_text("\n".join(sorted(holdout)) + "\n", encoding="utf-8")
    from collections import Counter

    print(f"test set: {len(rows)} clips, {sum(float(r['duration']) for r in rows) / 60:.1f} min, "
          f"{len(per_ep)} episodes; held out of training: {len(holdout)} clips "
          f"({sum(float(x['duration']) for x in train if x['id'] in holdout) / 60:.1f} min)")
    print("  emotions:", dict(Counter(r["emotion"] for r in rows)), " types:", dict(Counter(r["type"] for r in rows)))
    print("  arousal terciles:", dict(Counter(0 if level[r["id"]] < t1 else 1 if level[r["id"]] < t2 else 2 for r in test)))
    print(f"  reference: same emotion {np.mean([r['emotion'] == r['ref_emotion'] for r in rows]) * 100:.0f}%, "
          f"arousal distance median {np.median([r['ref_arousal_dist'] for r in rows]):.2f}")
    print(f"-> {out / 'testset.csv'}")
