"""
08_holdout_check.py -- score ONLY the clips that were held out untouched
by 07_expand_manifest.py. This is your honest before/after number.

Usage:
    python 08_holdout_check.py --holdout held_out_final.csv --model model_v2.pt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import librosa

import importlib.util
spec = importlib.util.spec_from_file_location("train03", Path(__file__).parent / "03_train.py")
train03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train03)

from risk_bands import classify


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="held_out_final.csv", type=Path)
    ap.add_argument("--model", default="model.pt", type=Path)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train03.build_model(pretrained=False).to(device)
    ck = torch.load(args.model, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()

    df = pd.read_csv(args.holdout)
    print(f"{'file':<28}{'truth':<6}{'score':<8}{'band':<11}{'correct?'}")
    print("-" * 65)
    n_correct, n_total, n_unc = 0, 0, 0
    for _, row in df.iterrows():
        score = score_file(row["path"], model, device)
        rv = classify(score)
        truth = "FAKE" if row["label"] == 1 else "REAL"
        if rv.band == "UNCERTAIN":
            n_unc += 1
            status = "--"
        else:
            verdict = "FAKE" if rv.band == "SYNTHETIC" else "REAL"
            correct = verdict == truth
            n_total += 1
            n_correct += int(correct)
            status = "YES" if correct else "NO - MISS"
        print(f"{Path(row['path']).name:<28}{truth:<6}{score:<8.3f}"
              f"{rv.band:<11}{status}")

    print("-" * 65)
    if n_total:
        print(f"Confident accuracy: {n_correct}/{n_total} ({100*n_correct/n_total:.1f}%)")
    print(f"Uncertain (escalated): {n_unc}/{len(df)}")


if __name__ == "__main__":
    main()
