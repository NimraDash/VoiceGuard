"""
01_audit.py  -- RUN THIS BEFORE YOU TRAIN ANYTHING.

Purpose: find out whether your real/fake folders are separable by trivial
metadata (sample rate, duration, loudness, file format). If they are, any
model you train is learning the shortcut, not the vocoder artifacts, and it
will fail on a real call.

Usage:
    python 01_audit.py --real /path/to/real --fake /path/to/fake

Reads:  your two folders
Writes: audit_report.txt, file_index.csv
"""

import argparse
import os
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

warnings.filterwarnings("ignore")

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus"}


def list_audio(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXT]


def probe(path: Path):
    """Cheap metadata probe. Does not decode the whole file where possible."""
    row = {
        "path": str(path),
        "name": path.name,
        "ext": path.suffix.lower(),
        "bytes": path.stat().st_size,
    }
    try:
        info = sf.info(str(path))
        row["sr"] = info.samplerate
        row["duration"] = info.frames / info.samplerate if info.samplerate else np.nan
        row["channels"] = info.channels
        row["subtype"] = info.subtype
    except Exception:
        # mp3/m4a may need a decode; fall back to librosa
        try:
            import librosa
            y, sr = librosa.load(str(path), sr=None, mono=True)
            row["sr"] = sr
            row["duration"] = len(y) / sr
            row["channels"] = 1
            row["subtype"] = "DECODED"
        except Exception as e:
            row["sr"] = np.nan
            row["duration"] = np.nan
            row["channels"] = np.nan
            row["subtype"] = f"ERR:{type(e).__name__}"
    return row


def loudness_stats(path: Path, max_seconds=6.0):
    """RMS and peak on a short head of the file. Cheap but very revealing."""
    try:
        import librosa
        y, sr = librosa.load(str(path), sr=16000, mono=True, duration=max_seconds)
        if y.size == 0:
            return np.nan, np.nan, np.nan
        rms = float(np.sqrt(np.mean(y ** 2)))
        peak = float(np.max(np.abs(y)))
        # fraction of samples that are near-digital-silence: TTS is often
        # unnaturally clean here, real mic recordings are not
        silent_frac = float(np.mean(np.abs(y) < 1e-4))
        return rms, peak, silent_frac
    except Exception:
        return np.nan, np.nan, np.nan


def shortcut_auc(df: pd.DataFrame, cols):
    """Train a trivial classifier on metadata ONLY. High AUC = your dataset leaks."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    sub = df[cols + ["label"]].dropna(subset=["label"])
    X = sub[cols].values
    y = sub["label"].values
    if len(np.unique(y)) < 2:
        return np.nan
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )
    proba = cross_val_predict(pipe, X, y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def filename_prefixes(names, n=12):
    """Guess speaker / generator groupings from filename structure."""
    seps = ["_", "-", "."]
    pref = []
    for nm in names:
        stem = Path(nm).stem
        cut = len(stem)
        for s in seps:
            if s in stem:
                cut = min(cut, stem.index(s))
        pref.append(stem[:cut] if cut > 0 else stem)
    return Counter(pref).most_common(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, type=Path)
    ap.add_argument("--fake", required=True, type=Path)
    ap.add_argument("--out", default=Path("."), type=Path)
    ap.add_argument("--loudness-sample", type=int, default=600,
                    help="how many files per class to decode for loudness stats")
    args = ap.parse_args()

    rows = []
    for root, label, tag in [(args.real, 0, "real"), (args.fake, 1, "fake")]:
        files = list_audio(root)
        print(f"[{tag}] found {len(files)} audio files under {root}")
        for p in files:
            r = probe(p)
            r["label"] = label
            r["class"] = tag
            rows.append(r)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No audio files found. Check your paths.")

    # loudness on a random subset (decoding everything is slow)
    rng = np.random.default_rng(0)
    df["rms"] = np.nan
    df["peak"] = np.nan
    df["silent_frac"] = np.nan
    for tag in ["real", "fake"]:
        idx = df.index[df["class"] == tag].to_numpy()
        pick = rng.choice(idx, size=min(args.loudness_sample, len(idx)), replace=False)
        for i in pick:
            rms, peak, sfr = loudness_stats(Path(df.at[i, "path"]))
            df.at[i, "rms"] = rms
            df.at[i, "peak"] = peak
            df.at[i, "silent_frac"] = sfr

    lines = []
    def w(s=""):
        print(s)
        lines.append(str(s))

    w("=" * 68)
    w("DATASET AUDIT")
    w("=" * 68)
    w(f"total files: {len(df)}   real: {(df.label==0).sum()}   fake: {(df.label==1).sum()}")
    w()

    w("--- 1. SAMPLE RATE (the #1 leak) ---")
    w(pd.crosstab(df["class"], df["sr"]).to_string())
    w()

    w("--- 2. FILE FORMAT / SUBTYPE (leak #2) ---")
    w(pd.crosstab(df["class"], df["ext"]).to_string())
    w(pd.crosstab(df["class"], df["subtype"]).to_string())
    w()

    w("--- 3. DURATION ---")
    w(df.groupby("class")["duration"].describe().to_string())
    w()

    w("--- 4. LOUDNESS / SILENCE (on sampled subset) ---")
    w(df.groupby("class")[["rms", "peak", "silent_frac"]].mean().to_string())
    w()

    w("--- 5. FILENAME STRUCTURE (for grouped splitting) ---")
    for tag in ["real", "fake"]:
        w(f"  {tag} top prefixes: {filename_prefixes(df.loc[df['class']==tag,'name'])}")
    w()

    w("--- 6. SHORTCUT TEST: can metadata alone predict the label? ---")
    auc_all = shortcut_auc(df, ["sr", "duration", "bytes"])
    auc_loud = shortcut_auc(df.dropna(subset=["rms"]), ["rms", "peak", "silent_frac"])
    w(f"  AUC from (sample_rate, duration, filesize) : {auc_all:.3f}")
    w(f"  AUC from (rms, peak, silence_fraction)     : {auc_loud:.3f}")
    w()
    w("  INTERPRETATION")
    w("    AUC ~0.50  -> clean, no metadata shortcut")
    w("    AUC  0.65+ -> a real shortcut exists, you MUST normalise it away")
    w("    AUC  0.90+ -> severe. Your dataset is separable without listening.")
    w()

    verdicts = []
    if not np.isnan(auc_all) and auc_all > 0.65:
        verdicts.append("METADATA LEAK: resample everything to 16 kHz and "
                        "random-crop to a fixed duration before training.")
    if not np.isnan(auc_loud) and auc_loud > 0.65:
        verdicts.append("LOUDNESS LEAK: peak-normalise every clip, and add "
                        "random gain augmentation.")
    if df.loc[df['class']=='real','sr'].mode().tolist() != df.loc[df['class']=='fake','sr'].mode().tolist():
        verdicts.append("Sample rates differ between classes. Force a common SR.")
    if not verdicts:
        verdicts.append("No obvious shortcut found. Proceed, but still resample "
                        "and peak-normalise as a matter of hygiene.")

    w("--- 7. VERDICT / REQUIRED ACTIONS ---")
    for v in verdicts:
        w(f"  * {v}")
    w()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit_report.txt").write_text("\n".join(lines))
    df.to_csv(args.out / "file_index.csv", index=False)
    print(f"\nwrote {args.out/'audit_report.txt'} and {args.out/'file_index.csv'}")


if __name__ == "__main__":
    main()
