"""
09_calibrate.py -- set the three-band thresholds (LOW/HIGH) from real
evidence instead of a hand-picked guess. This does NOT retrain anything --
it only decides where to draw the GENUINE / UNCERTAIN / SYNTHETIC lines on
scores the model already produces.

Uses your val + test splits (NOT the tiny 8-clip holdout -- that sample is
too small to set thresholds from reliably, it's only for a final sanity
check).

Usage:
    python 09_calibrate.py --manifest manifest2.csv --model model_v2.pt
    python 09_calibrate.py --manifest manifest2.csv --model model_v2.pt --apply
      (--apply writes the new numbers directly into risk_bands.py)
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import librosa

import importlib.util
spec = importlib.util.spec_from_file_location("train03", Path(__file__).parent / "03_train.py")
train03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train03)

from risk_bands import classify, suggest_thresholds, LOW as CUR_LOW, HIGH as CUR_HIGH


def score_file(path, model, device):
    y, _ = librosa.load(str(path), sr=train03.SR, mono=True)
    if y.size < train03.CHUNK_LEN:
        y = np.pad(y, (0, train03.CHUNK_LEN - y.size))
    step = train03.SR * 2
    scores = []
    for start in range(0, max(1, len(y) - train03.CHUNK_LEN + 1), step):
        chunk = y[start:start + train03.CHUNK_LEN]
        if len(chunk) < train03.CHUNK_LEN:
            chunk = np.pad(chunk, (0, train03.CHUNK_LEN - len(chunk)))
        peak = np.max(np.abs(chunk)) + 1e-9
        chunk = (chunk / peak).astype(np.float32)
        x = train03.featurise(chunk)
        x = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.softmax(model(x), 1)[0, 1].item()
        scores.append(p)
    return float(np.mean(scores))


def evaluate_bands(labels, scores, low, high):
    n_correct, n_conf, n_unc = 0, 0, 0
    for lbl, sc in zip(labels, scores):
        rv = classify(sc, low, high)
        if rv.band == "UNCERTAIN":
            n_unc += 1
        else:
            n_conf += 1
            pred = 1 if rv.band == "SYNTHETIC" else 0
            n_correct += int(pred == lbl)
    acc = n_correct / n_conf if n_conf else float("nan")
    return acc, n_conf, n_unc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest2.csv", type=Path)
    ap.add_argument("--model", default="model_v2.pt", type=Path)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--target-fa", type=float, default=0.05,
                    help="target false-alarm / missed-fake rate for the "
                         "confident bands (0.05 = 5%%)")
    ap.add_argument("--apply", action="store_true",
                     help="write the new thresholds into risk_bands.py")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train03.build_model(pretrained=False).to(device)
    ck = torch.load(args.model, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()

    df = pd.read_csv(args.manifest)
    splits = set(s.strip() for s in args.splits.split(","))
    sub = df[df["split"].isin(splits)]
    print(f"scoring {len(sub)} clips from splits {splits} ...")

    real_scores, fake_scores, labels, scores = [], [], [], []
    for _, row in sub.iterrows():
        try:
            s = score_file(row["path"], model, device)
        except Exception as e:
            print(f"  skip {row['path']}: {e}")
            continue
        (real_scores if row["label"] == 0 else fake_scores).append(s)
        labels.append(int(row["label"]))
        scores.append(s)

    print(f"scored: {len(real_scores)} real, {len(fake_scores)} fake\n")

    new_low, new_high = suggest_thresholds(real_scores, fake_scores,
                                           target_false_alarm=args.target_fa)

    print(f"CURRENT thresholds : LOW={CUR_LOW:.3f}  HIGH={CUR_HIGH:.3f}")
    print(f"SUGGESTED thresholds: LOW={new_low:.3f}  HIGH={new_high:.3f}\n")

    acc_old, conf_old, unc_old = evaluate_bands(labels, scores, CUR_LOW, CUR_HIGH)
    acc_new, conf_new, unc_new = evaluate_bands(labels, scores, new_low, new_high)

    print("Comparison on this same val+test data:")
    print(f"  current thresholds -> confident-accuracy {acc_old:.3f}  "
          f"({conf_old} confident / {unc_old} uncertain)")
    print(f"  suggested thresholds -> confident-accuracy {acc_new:.3f}  "
          f"({conf_new} confident / {unc_new} uncertain)")

    if args.apply:
        rb_path = Path(__file__).parent / "risk_bands.py"
        src = rb_path.read_text()
        src = re.sub(r"^LOW = [0-9.]+.*$",
                     f"LOW = {new_low:.4f}    # calibrated by 09_calibrate.py",
                     src, count=1, flags=re.MULTILINE)
        src = re.sub(r"^HIGH = [0-9.]+.*$",
                     f"HIGH = {new_high:.4f}   # calibrated by 09_calibrate.py",
                     src, count=1, flags=re.MULTILINE)
        rb_path.write_text(src)
        print(f"\napplied -> risk_bands.py now uses LOW={new_low:.4f} "
              f"HIGH={new_high:.4f}")
    else:
        print("\n(dry run -- add --apply to write these into risk_bands.py)")


if __name__ == "__main__":
    main()
