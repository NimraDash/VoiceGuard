"""
07_expand_manifest.py -- fold your fresh recordings into training, while
keeping a few completely untouched for an honest before/after check.

Keeps your ORIGINAL train/val/test split exactly as it was (so you can
still compare apples to apples), and adds the new clips mostly into train,
a few into val, and reserves a small holdout that touches NOTHING -- those
files are never used for training or threshold selection, only for the
final sanity check after retraining.

Usage:
    python 07_expand_manifest.py --manifest manifest.csv \
        --new-real New_Real --new-fake New_Fake \
        --holdout-per-class 4
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


def probe_duration(path: Path):
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate, info.samplerate
    except Exception:
        return np.nan, np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv", type=Path)
    ap.add_argument("--new-real", required=True, type=Path)
    ap.add_argument("--new-fake", required=True, type=Path)
    ap.add_argument("--holdout-per-class", type=int, default=4)
    ap.add_argument("--val-frac-new", type=float, default=0.2)
    ap.add_argument("--out", default="manifest2.csv", type=Path)
    ap.add_argument("--holdout-out", default="held_out_final.csv", type=Path)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    orig = pd.read_csv(args.manifest)
    rng = np.random.default_rng(args.seed)

    new_rows = []
    for root, label, tag in [(args.new_real, 0, "real"), (args.new_fake, 1, "fake")]:
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXT)
        rng.shuffle(files)
        for p in files:
            dur, sr = probe_duration(p)
            new_rows.append({
                "path": str(p), "label": label, "class": tag,
                "group": f"newfile:{p.stem}", "generator": "unknown" if label else "bonafide",
                "sr": sr, "duration": dur,
            })

    new_df = pd.DataFrame(new_rows)
    print(f"new clips found: {len(new_df)}  "
          f"(real {sum(new_df['class']=='real')}, fake {sum(new_df['class']=='fake')})")

    # ---- reserve an untouched holdout, split by class ----------------------
    holdout_parts, remain_parts = [], []
    for tag in ["real", "fake"]:
        sub = new_df[new_df["class"] == tag].reset_index(drop=True)
        k = min(args.holdout_per_class, len(sub))
        holdout_idx = rng.choice(len(sub), size=k, replace=False)
        mask = np.zeros(len(sub), dtype=bool)
        mask[holdout_idx] = True
        holdout_parts.append(sub[mask])
        remain_parts.append(sub[~mask])

    holdout_df = pd.concat(holdout_parts, ignore_index=True)
    remain_df = pd.concat(remain_parts, ignore_index=True)

    # ---- split the remainder into train/val ---------------------------------
    remain_df = remain_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_val = int(len(remain_df) * args.val_frac_new)
    remain_df["split"] = "train"
    remain_df.loc[:n_val - 1, "split"] = "val"

    combined = pd.concat([orig, remain_df], ignore_index=True)
    combined = combined[["path", "label", "class", "group", "generator",
                         "split", "sr", "duration"]]
    combined.to_csv(args.out, index=False)
    holdout_df[["path", "label", "class"]].to_csv(args.holdout_out, index=False)

    print("\ncombined split sizes:")
    print(pd.crosstab(combined["split"], combined["class"]).to_string())
    print(f"\nheld out untouched (final honest check): {len(holdout_df)} clips")
    print(f"  -> {args.holdout_out}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
