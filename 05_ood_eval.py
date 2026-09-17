"""
05_ood_eval.py -- batch out-of-distribution test.

Point this at a NEW folder of real and fake clips (not used in training) and
it tells you, file by file and overall, whether the model generalises.

Usage:
    python 05_ood_eval.py --real New_Real --fake New_Fake --model model.pt
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

from risk_bands import classify

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


def score_file(path, model, device):
    """Return the average P(fake) across all 4s chunks in the file."""
    try:
        y, _ = librosa.load(str(path), sr=train03.SR, mono=True)
    except Exception as e:
        return None, f"COULD NOT READ: {e}"
    if y.size < train03.SR * 0.5:
        return None, "TOO SHORT (<0.5s)"
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
    return float(np.mean(scores)), None


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
    thr = ck["val"].get("thr", 0.5)
    print(f"loaded {args.model}  (training val EER was {ck['val'].get('eer', 'n/a'):.3f}, "
          f"threshold {thr:.3f})\n")

    rows = []
    for root, label, tag in [(args.real, 0, "REAL"), (args.fake, 1, "FAKE")]:
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXT)
        for p in files:
            score, err = score_file(p, model, device)
            rows.append((p.name, tag, label, score, err))

    print(f"{'file':<30} {'truth':<6} {'P(fake)':<9} {'band':<11} {'verdict':<12} {'correct?'}")
    print("-" * 85)
    n_correct, n_total = 0, 0
    n_uncertain = 0
    fake_scores, real_scores = [], []
    for name, tag, label, score, err in rows:
        if err:
            print(f"{name:<30} {tag:<6} {'--':<9} {err}")
            continue
        rv = classify(score)
        verdict = "FAKE" if rv.band == "SYNTHETIC" else ("REAL" if rv.band == "GENUINE" else "UNSURE")
        if rv.band == "UNCERTAIN":
            n_uncertain += 1
            correct = None
        else:
            correct = (verdict == tag)
            n_total += 1
            n_correct += int(correct)
        (fake_scores if label == 1 else real_scores).append(score)
        correct_str = "--" if correct is None else ("YES" if correct else "NO - MISS")
        print(f"{name:<30} {tag:<6} {score:<9.3f} {rv.band:<11} {verdict:<12} {correct_str}")

    print("-" * 85)
    if n_total:
        print(f"\nAccuracy on CONFIDENT calls only: {n_correct}/{n_total} "
              f"({100*n_correct/n_total:.1f}%)")
    print(f"Flagged UNCERTAIN (escalate to human verification): {n_uncertain} "
          f"of {len(rows)} clips")
    if real_scores:
        print(f"Real clips  -> mean P(fake) = {np.mean(real_scores):.3f}  "
              f"(want this LOW, near 0)")
    if fake_scores:
        print(f"Fake clips  -> mean P(fake) = {np.mean(fake_scores):.3f}  "
              f"(want this HIGH, near 1)")

    if real_scores and fake_scores:
        gap = np.mean(fake_scores) - np.mean(real_scores)
        print(f"\nSeparation (fake avg - real avg): {gap:+.3f}")
        if gap > 0.3:
            print("-> Model is genuinely separating fresh real vs fake. Good sign.")
        elif gap > 0.05:
            print("-> Weak separation. Model has some signal but is not reliable "
                  "on unseen material. Report this honestly as a limitation.")
        else:
            print("-> No real separation on fresh data. The model is likely "
                  "overfit to your original training set's specific sources. "
                  "This is a real and common failure mode -- report it as your "
                  "'why we need abstention / human verification' finding, not "
                  "as a hidden flaw.")


if __name__ == "__main__":
    main()
