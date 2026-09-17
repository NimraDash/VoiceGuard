"""
02_manifest.py -- build train/val/test splits that do not leak.

Two rules enforced here:
  1. GROUPED SPLIT. All clips sharing a speaker/source prefix go to the same
     split. Otherwise the model memorises voices, not artifacts.
  2. HELD-OUT GENERATOR. If the fake folder has subfolders (one per TTS /
     vocoder method), one whole method is reserved for the test set only.
     This is your leave-one-generator-out evaluation, the number that proves
     you actually measured generalisation.

Usage:
    python 02_manifest.py --index file_index.csv --holdout-generator auto
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


GENERIC_FOLDER_TOKENS = {
    "real", "fake", "wav", "audio", "clips",
    "real_audios", "fake_audios", "realaudios", "fakeaudios",
    "genuine", "spoof", "bonafide", "synthetic", "generated",
}


def group_key(path_str: str) -> str:
    """
    Derive a grouping key. Priority:
      1. first subdirectory under the class folder (usually speaker or method),
         UNLESS that folder name is just a generic class label like
         "Real_Audios" / "Fake_Audios" -- that carries no speaker info and
         would collapse the whole class into one group.
      2. leading token of the filename, UNLESS that token is just "real" or
         "fake" (same problem, one level down) -- e.g. "real_001.wav" gives
         nothing useful, but "LJ001_0003.wav" gives "LJ001".
      3. fallback: the filename itself. This means NO grouping protection --
         every clip is treated as independent. Caller is told this happened.
    """
    p = Path(path_str)
    parts = p.parts

    if len(parts) >= 2:
        parent = parts[-2]
        if parent.lower().replace(" ", "_") not in GENERIC_FOLDER_TOKENS:
            return f"dir:{parent}"

    stem = p.stem
    m = re.match(r"^([A-Za-z]*\d+|[A-Za-z]+)", stem)
    token = m.group(1) if m else stem
    if token.lower() not in GENERIC_FOLDER_TOKENS:
        return f"name:{token}"

    return f"file:{stem}"


def generator_key(path_str: str) -> str:
    """Subfolder under the fake root, if any. Used for leave-one-out."""
    p = Path(path_str)
    parts = [x.lower() for x in p.parts]
    for anchor in ("fake", "spoof", "synthetic", "generated"):
        if anchor in parts:
            i = parts.index(anchor)
            if i + 1 < len(parts) - 1:
                return p.parts[i + 1]
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="file_index.csv", type=Path)
    ap.add_argument("--out", default="manifest.csv", type=Path)
    ap.add_argument("--holdout-generator", default="auto",
                    help="'auto' picks the smallest non-unknown generator, "
                         "'none' disables, or give an exact name")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    df = df.dropna(subset=["duration"])
    df = df[df["duration"] >= 1.0].reset_index(drop=True)

    df["group"] = df["path"].map(group_key)
    df["generator"] = np.where(df["label"] == 1,
                               df["path"].map(generator_key), "bonafide")

    print("groups found:", df["group"].nunique())
    print(df.groupby("class")["group"].nunique().to_string())

    frac_no_grouping = df["group"].str.startswith("file:").mean()
    if frac_no_grouping > 0.5:
        print(f"\n  WARNING: {frac_no_grouping:.0%} of files fell back to "
              f"per-file grouping (no speaker/source ID recoverable from "
              f"filenames or folders). The split below is a plain random "
              f"split, NOT speaker-grouped. This is honest to disclose: "
              f"say in your slides that speaker-level generalisation could "
              f"not be measured because the dataset provides no speaker "
              f"labels, and that your unseen-generator / fresh-clone test "
              f"is doing that job instead.\n")
    print("\ngenerators found:")
    print(df["generator"].value_counts().to_string())

    # ---- leave-one-generator-out -------------------------------------------
    hog = args.holdout_generator
    gens = [g for g in df["generator"].unique() if g not in ("bonafide", "unknown")]
    if hog == "auto":
        hog = None
        if len(gens) >= 2:
            counts = df[df["generator"].isin(gens)]["generator"].value_counts()
            # pick the smallest generator that still has enough samples
            viable = counts[counts >= 100]
            hog = viable.index[-1] if len(viable) >= 2 else None
    elif hog == "none":
        hog = None

    df["split"] = ""
    if hog:
        print(f"\n>>> HELD-OUT GENERATOR: {hog}  "
              f"({(df['generator']==hog).sum()} clips) -> test only")
        df.loc[df["generator"] == hog, "split"] = "test_unseen_gen"
    else:
        print("\n>>> No generator subfolders usable. "
              "Leave-one-generator-out is NOT possible with this dataset. "
              "Say so honestly in your slides, and substitute an "
              "out-of-distribution test with a clone you generate yourself.")

    # ---- grouped random split on the remainder ------------------------------
    rest = df[df["split"] == ""].copy()
    rng = np.random.default_rng(args.seed)
    groups = np.array(sorted(rest["group"].unique()), dtype=object)
    rng.shuffle(groups)

    n = len(groups)
    n_test = int(n * args.test_frac)
    n_val = int(n * args.val_frac)
    test_g = set(groups[:n_test])
    val_g = set(groups[n_test:n_test + n_val])

    def assign(g):
        if g in test_g:
            return "test"
        if g in val_g:
            return "val"
        return "train"

    df.loc[rest.index, "split"] = rest["group"].map(assign)

    print("\nsplit sizes:")
    print(pd.crosstab(df["split"], df["class"]).to_string())

    # ---- sanity checks you must not skip -----------------------------------
    overlap = set(df[df.split == "train"]["group"]) & set(df[df.split == "test"]["group"])
    assert not overlap, f"GROUP LEAK between train and test: {list(overlap)[:5]}"

    for s in df["split"].unique():
        sub = df[df["split"] == s]
        if sub["label"].nunique() < 2 and s != "test_unseen_gen":
            print(f"  WARNING: split '{s}' has only one class.")

    df[["path", "label", "class", "group", "generator", "split",
        "sr", "duration"]].to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
