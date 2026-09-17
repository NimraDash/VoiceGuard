"""
03_train.py -- 3-channel spectro-image classifier with telephony-realistic
augmentation.

Channels fed to the CNN:
    ch0  log-mel spectrogram        (what everyone does)
    ch1  modified group delay       (phase artifacts; survives codecs)
    ch2  delta of log-mel           (micro-prosody / temporal variation)

Augmentation applied at TRAIN time only, with probability, so the model is
forced to find artifacts that survive a phone line:
    - resample to 8 kHz and back        (telephony band limit)
    - mu-law quantise                   (G.711, what actual calls use)
    - random gain                       (kills loudness shortcut)
    - gaussian noise                    (channel noise)

Usage:
    python 03_train.py --manifest manifest.csv --epochs 8
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import librosa

SR = 16000
CHUNK_SEC = 4.0
CHUNK_LEN = int(SR * CHUNK_SEC)
N_MELS = 80
N_FFT = 512
HOP = 160  # 10 ms


# ----------------------------------------------------------------------------
# augmentation
# ----------------------------------------------------------------------------
def mu_law_roundtrip(y, mu=255):
    y = np.clip(y, -1.0, 1.0)
    enc = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    q = np.round((enc + 1) / 2 * mu).astype(np.int16)
    dec = (q.astype(np.float32) / mu) * 2 - 1
    return np.sign(dec) * (1.0 / mu) * ((1 + mu) ** np.abs(dec) - 1)


def voice_diversity_aug(y, rng):
    """
    Pitch/speed perturbation. Forces the model to key on synthesis
    artifacts rather than one specific speaker's exact pitch/timing --
    the thing that made it over-rely on Nimra's specific voiceprint.
    """
    if rng.random() < 0.5:
        semitones = rng.uniform(-2.0, 2.0)
        try:
            y = librosa.effects.pitch_shift(y, sr=SR, n_steps=semitones)
        except Exception:
            pass
    if rng.random() < 0.5:
        rate = rng.uniform(0.9, 1.1)
        try:
            y2 = librosa.effects.time_stretch(y, rate=rate)
            # keep chunk length consistent; caller will pad/crop again
            y = y2
        except Exception:
            pass
    return y.astype(np.float32)


def telephony_aug(y, rng):
    y = voice_diversity_aug(y, rng)
    if rng.random() < 0.7:  # band limit
        y = librosa.resample(y, orig_sr=SR, target_sr=8000)
        y = librosa.resample(y, orig_sr=8000, target_sr=SR)
    if rng.random() < 0.5:  # G.711
        y = mu_law_roundtrip(y)
    if rng.random() < 0.8:  # gain
        y = y * float(rng.uniform(0.3, 1.2))
    if rng.random() < 0.4:  # noise
        snr = rng.uniform(15, 40)
        p = np.mean(y ** 2) + 1e-12
        y = y + rng.normal(0, math.sqrt(p / (10 ** (snr / 10))), y.shape)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------
_MEL_FB = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)


def modified_group_delay(S_complex, alpha=0.4, gamma=0.9):
    """Cheap MGD: derivative of unwrapped phase along frequency, smoothed."""
    phase = np.angle(S_complex)
    unwrapped = np.unwrap(phase, axis=0)
    gd = -np.diff(unwrapped, axis=0, prepend=unwrapped[:1])
    mag = np.abs(S_complex) + 1e-10
    smoothed = mag ** gamma
    mgd = gd / (smoothed + 1e-10)
    mgd = np.sign(mgd) * (np.abs(mgd) ** alpha)
    return mgd


def featurise(y):
    """-> float32 array (3, N_MELS, T)"""
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP, win_length=400)
    mag = np.abs(S)

    mel = _MEL_FB @ (mag ** 2)
    logmel = librosa.power_to_db(mel + 1e-10)

    mgd = modified_group_delay(S)
    mgd_mel = _MEL_FB @ np.abs(mgd)
    mgd_mel = np.log1p(mgd_mel)

    dmel = librosa.feature.delta(logmel, width=5)

    def norm(x):
        return (x - x.mean()) / (x.std() + 1e-6)

    return np.stack([norm(logmel), norm(mgd_mel), norm(dmel)]).astype(np.float32)


# ----------------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------------
class SpoofChunks(Dataset):
    def __init__(self, df, train=True, seed=0):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def _load(self, path):
        y, _ = librosa.load(path, sr=SR, mono=True)
        if y.size < CHUNK_LEN:
            y = np.pad(y, (0, CHUNK_LEN - y.size))
        if self.train:
            start = self.rng.integers(0, y.size - CHUNK_LEN + 1)
        else:
            start = max(0, (y.size - CHUNK_LEN) // 2)  # deterministic for eval
        y = y[start:start + CHUNK_LEN]
        peak = np.max(np.abs(y)) + 1e-9
        return (y / peak).astype(np.float32)   # peak-normalise: kills loudness leak

    def __getitem__(self, i):
        row = self.df.iloc[i]
        try:
            y = self._load(row["path"])
        except Exception:
            y = np.zeros(CHUNK_LEN, dtype=np.float32)
        if self.train:
            y = telephony_aug(y, self.rng)
            # pitch/time-stretch can change length -- restore fixed length
            if y.size < CHUNK_LEN:
                y = np.pad(y, (0, CHUNK_LEN - y.size))
            elif y.size > CHUNK_LEN:
                y = y[:CHUNK_LEN]
        x = featurise(y)
        return torch.from_numpy(x), torch.tensor(int(row["label"]), dtype=torch.long)


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
def build_model(pretrained: bool = True):
    """
    pretrained=True  -> used during TRAINING, downloads ImageNet weights
                        as a starting point (needs internet once, then cached).
    pretrained=False -> used for INFERENCE (predict/eval/app scripts). The
                        architecture is identical either way; we're about to
                        overwrite every weight with our own checkpoint, so
                        there is no reason to touch the network and no reason
                        for a live demo to depend on internet access.
    """
    try:
        import timm
        return timm.create_model("efficientnet_b0", pretrained=pretrained,
                                 in_chans=3, num_classes=2)
    except Exception:
        from torchvision.models import resnet18, ResNet18_Weights
        w = ResNet18_Weights.DEFAULT if pretrained else None
        m = resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def compute_eer(labels, scores):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    S, Y = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), 1)[:, 1]
        S.append(p.cpu().numpy())
        Y.append(y.numpy())
    S = np.concatenate(S)
    Y = np.concatenate(Y)
    if len(np.unique(Y)) < 2:
        return {"n": len(Y), "mean_score": float(S.mean())}
    from sklearn.metrics import roc_auc_score
    eer, thr = compute_eer(Y, S)
    return {"n": len(Y), "auc": float(roc_auc_score(Y, S)),
            "eer": eer, "thr": thr, "acc": float(((S > 0.5) == Y).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv", type=Path)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="model.pt", type=Path)
    ap.add_argument("--resume", default=None, type=Path,
                    help="path to an existing model.pt to fine-tune from, "
                         "instead of starting from ImageNet weights")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    df = pd.read_csv(args.manifest)
    tr = df[df.split == "train"]
    va = df[df.split == "val"]
    te = df[df.split == "test"]
    ug = df[df.split == "test_unseen_gen"]
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}  unseen-gen {len(ug)}")

    mk = lambda d, t, s=0: DataLoader(SpoofChunks(d, t, s), batch_size=args.bs,
                                      shuffle=t, num_workers=args.workers,
                                      drop_last=t, pin_memory=True)
    tl, vl = mk(tr, True), mk(va, False)
    testl = mk(te, False)
    ugl = mk(ug, False) if len(ug) else None

    model = build_model().to(device)
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        print(f"resumed weights from {args.resume}  "
              f"(its val was: {ck.get('val')})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.epochs * len(tl)))

    w = torch.tensor([1.0, len(tr[tr.label == 0]) / max(1, len(tr[tr.label == 1]))],
                     dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)

    best = 1e9
    for ep in range(args.epochs):
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        for bi, (x, y) in enumerate(tl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * y.size(0)
            seen += y.size(0)
            if bi % 50 == 0:
                print(f"  ep{ep} step{bi}/{len(tl)} loss {tot/max(1,seen):.4f}")
        m = evaluate(model, vl, device)
        print(f"[epoch {ep}] train_loss {tot/max(1,seen):.4f}  val {m}  "
              f"({time.time()-t0:.0f}s)")
        if m.get("eer", 1.0) < best:
            best = m.get("eer", 1.0)
            torch.save({"model": model.state_dict(), "val": m}, args.out)
            print(f"  saved -> {args.out}")

    print("\n=== FINAL (best checkpoint) ===")
    ck = torch.load(args.out, map_location=device)
    model.load_state_dict(ck["model"])
    print("val            :", ck["val"])
    print("test (seen)    :", evaluate(model, testl, device))
    if ugl:
        print("test UNSEEN GEN:", evaluate(model, ugl, device))
        print("\n^ THE GAP BETWEEN THOSE TWO LINES IS YOUR PRESENTATION.")


if __name__ == "__main__":
    main()
