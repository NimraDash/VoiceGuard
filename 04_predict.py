"""
04_predict.py -- run the trained model on ANY single audio file.

Use this for the out-of-distribution test: point it at a fresh synthetic
clip (made with a tool NOT in your training data) and at a few real clips,
and see what probability comes out.

Usage:
    python 04_predict.py --audio path/to/clip.wav
    python 04_predict.py --audio path/to/clip.wav --model model.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import librosa

import importlib.util
spec = importlib.util.spec_from_file_location("train03", Path(__file__).parent / "03_train.py")
train03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train03)


def predict_file(path, model, device):
    y, _ = librosa.load(path, sr=train03.SR, mono=True)
    if y.size < train03.CHUNK_LEN:
        y = np.pad(y, (0, train03.CHUNK_LEN - y.size))

    n_chunks = max(1, (len(y) - train03.CHUNK_LEN) // (train03.SR * 2) + 1)
    scores = []
    step = train03.SR * 2  # 2 second hop between chunk starts
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
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--model", default="model.pt", type=Path)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train03.build_model(pretrained=False).to(device)
    ck = torch.load(args.model, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()

    print(f"loaded model.pt  (its own val EER was: {ck['val'].get('eer', 'n/a')})")
    print(f"analysing: {args.audio}")

    scores = predict_file(str(args.audio), model, device)
    print(f"\nper-chunk P(fake): {[round(s, 3) for s in scores]}")
    avg = float(np.mean(scores))
    print(f"average P(fake)  : {avg:.3f}")

    thr = ck["val"].get("thr", 0.5)
    verdict = "SYNTHETIC / FAKE" if avg > thr else "GENUINE"
    print(f"verdict (thr={thr:.2f}): {verdict}")


if __name__ == "__main__":
    main()
