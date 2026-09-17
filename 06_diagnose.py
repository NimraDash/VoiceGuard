"""
06_diagnose.py -- find out WHY specific clips are being missed, before you
spend time retraining blind.

Checks the two most common causes of "confidently wrong" predictions:
  1. Padding artifact -- very short clips get zero-padded up to 4 seconds;
     if the model learned "lots of silence padding = real", short fakes
     will be misclassified as real regardless of content.
  2. Loudness / spectral tilt -- despite peak-normalisation, some other
     property (noise floor, spectral flatness) may still correlate with
     the label by accident.

Usage:
    python 06_diagnose.py --real New_Real --fake New_Fake --model model.pt
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

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


def analyse_file(path, model, device):
    y, sr0 = librosa.load(str(path), sr=None, mono=True)
    duration = len(y) / sr0
    y16, _ = librosa.load(str(path), sr=train03.SR, mono=True)

    orig_len = len(y16)
    pad_frac = max(0.0, (train03.CHUNK_LEN - orig_len) / train03.CHUNK_LEN)

    if orig_len < train03.CHUNK_LEN:
        y16 = np.pad(y16, (0, train03.CHUNK_LEN - orig_len))
    chunk = y16[:train03.CHUNK_LEN]
    peak = np.max(np.abs(chunk)) + 1e-9
    chunk_n = (chunk / peak).astype(np.float32)

    rms = float(np.sqrt(np.mean(chunk_n ** 2)))
    silent_frac = float(np.mean(np.abs(chunk_n) < 1e-4))

    x = train03.featurise(chunk_n)
    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(model(xt), 1)[0, 1].item()

    return {
        "duration": duration,
        "pad_frac": pad_frac,
        "rms": rms,
        "silent_frac": silent_frac,
        "score": p,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, type=Path)
    ap.add_argument("--fake", required=True, type=Path)
    ap.add_argument("--model", default="model.pt", type=Path)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train03.build_model(pretrained=False).to(device)
    ck = torch.load(args.model, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()

    rows = []
    for root, label, tag in [(args.real, 0, "REAL"), (args.fake, 1, "FAKE")]:
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXT)
        for p in files:
            try:
                r = analyse_file(p, model, device)
            except Exception as e:
                print(f"skip {p.name}: {e}")
                continue
            r["name"] = p.name
            r["label"] = label
            r["tag"] = tag
            rows.append(r)

    print(f"{'file':<28}{'truth':<6}{'dur(s)':<8}{'pad%':<7}{'rms':<8}"
          f"{'silent%':<9}{'score':<7}")
    print("-" * 75)
    for r in rows:
        print(f"{r['name']:<28}{r['tag']:<6}{r['duration']:<8.2f}"
              f"{100*r['pad_frac']:<7.1f}{r['rms']:<8.4f}"
              f"{100*r['silent_frac']:<9.1f}{r['score']:<7.3f}")

    # correlate padding with wrong-direction confident misses
    fake_rows = [r for r in rows if r["label"] == 1]
    confident_miss = [r for r in fake_rows if r["score"] < 0.05]
    other_fake = [r for r in fake_rows if r["score"] >= 0.05]

    print("\n--- DIAGNOSIS ---")
    if confident_miss:
        avg_pad_miss = np.mean([r["pad_frac"] for r in confident_miss])
        avg_pad_other = np.mean([r["pad_frac"] for r in other_fake]) if other_fake else 0
        avg_dur_miss = np.mean([r["duration"] for r in confident_miss])
        avg_dur_other = np.mean([r["duration"] for r in other_fake]) if other_fake else 0
        print(f"Confidently-wrong fakes (score<0.05): {len(confident_miss)}")
        print(f"  avg padding %%   : {100*avg_pad_miss:.1f}  "
              f"(vs {100*avg_pad_other:.1f}%% for correctly-flagged fakes)")
        print(f"  avg duration (s): {avg_dur_miss:.2f}  "
              f"(vs {avg_dur_other:.2f}s for correctly-flagged fakes)")
        if avg_pad_miss > avg_pad_other + 0.10 or avg_dur_miss < avg_dur_other - 1.0:
            print("\n  ==> LIKELY CAUSE: short/padded clips. The model has "
                  "learned to associate heavy silence-padding with 'real'. "
                  "FIX: change padding strategy, not full retrain scope.")
        else:
            print("\n  ==> Padding/duration does NOT explain the misses. "
                  "This is a genuine acoustic generalisation gap -- the "
                  "model's learned features don't transfer to this voice/"
                  "generator. FIX: needs more diverse training data, not "
                  "a padding tweak.")
    else:
        print("No confidently-wrong fakes in this run.")


if __name__ == "__main__":
    main()
