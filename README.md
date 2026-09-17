# VoiceGuard

Detection of AI-generated and cloned speech, with a cost-sensitive
escalation policy that turns a score into a recommended action.

---

## What this actually is

Three layers, in order:

1. **Detector.** Audio is cut into 4-second windows (2-second hop). Each
   window becomes a 3-channel image — log-mel spectrogram, modified group
   delay (phase artifacts), and delta-mel (frame-to-frame change) — and is
   classified by an ImageNet-pretrained EfficientNet-B0 fine-tuned on real
   and synthetic speech.

2. **Risk bands.** Window scores are averaged into one probability, then
   placed in one of three bands: genuine, uncertain, synthetic. The
   uncertain band exists because the detector is measurably less reliable on
   voices it has never encountered. Abstaining is treated as a valid output,
   not a failure.

3. **Escalation policy.** A reinforcement-learning policy chooses between
   keep listening / verify the caller / warn the operator / stop the
   transaction, using the score, its trend across windows, and the call
   context. See "About the RL layer" below for exactly what it does and does
   not claim.

---

## Setup

Requires Python 3.10 or 3.11.

```bash
pip install numpy pandas scikit-learn soundfile librosa torch torchvision timm flask
```

Files expected in the project folder:

```
Clone/
  Real_Audios/          original real clips
  Fake_Audios/          original synthetic clips
  New_Real/             fresh real clips (not from the original dataset)
  New_Fake/             fresh synthetic clips
  model_v2.pt           the trained detector
  rl_policy.json        the trained escalation policy
  *.py                  scripts below
  templates/index.html
  static/styles.css
  static/app.js
```

---

## Running the app

```bash
python app.py
```

Open `http://localhost:5000`.

**To demo on a phone:** the laptop and phone must be on the same wifi (a
laptop hotspot works and avoids venue-wifi problems entirely).

1. On the laptop run `ipconfig` and note the IPv4 address, e.g. `192.168.1.7`
2. On the phone open `http://192.168.1.7:5000`

Microphone recording requires a secure context. `localhost` counts as
secure; a plain-IP address over http does **not**, so the Record button may
be blocked on the phone. File upload always works. Test this before you
present — if recording is blocked, demo with uploaded clips.

---

## Rebuilding from scratch

Run in this order. Steps 1–3 are only needed if you retrain the detector.

| Step | Command | Purpose |
| --- | --- | --- |
| 1 | `python 01_audit.py --real Real_Audios --fake Fake_Audios` | Detect dataset shortcuts before trusting any result |
| 2 | `python 02_manifest.py --index file_index.csv` | Build leakage-aware train/val/test splits |
| 3 | `python 03_train.py --manifest manifest.csv --epochs 8` | Train the detector |
| 4 | `python 07_expand_manifest.py --manifest manifest.csv --new-real New_Real --new-fake New_Fake --holdout-per-class 4` | Fold in fresh voices, reserve an untouched holdout |
| 5 | `python 03_train.py --manifest manifest2.csv --epochs 15 --resume model.pt --out model_v2.pt` | Fine-tune on the expanded set |
| 6 | `python 08_holdout_check.py --holdout held_out_final.csv --model model_v2.pt` | Score the untouched holdout — the honest number |
| 7 | `python 09_calibrate.py --manifest manifest2.csv --model model_v2.pt --apply` | Set band thresholds from data, not by hand |
| 8 | `python 10_train_rl.py --episodes 250000` | Train the escalation policy, compare against baseline |

Other tools:

- `python 04_predict.py --audio clip.wav` — score one file
- `python 05_ood_eval.py --real New_Real --fake New_Fake` — batch evaluation
- `python 06_diagnose.py --real New_Real --fake New_Fake` — investigate failures

---

## About the RL layer

**What it does.** A tabular Q-learning policy picks an action at each
window: WAIT, CHALLENGE, ALERT, BLOCK. State is the running mean score, its
trend, how far into the call we are, whether the caller is a known contact,
and the transaction value at stake.

**Why RL rather than a threshold.** A fixed cutoff is optimal when you
decide once. Here the decision repeats every few seconds, waiting carries a
growing cost, challenging carries friction, and the right move depends on
whether the score is rising or falling. That is a sequential decision under
asymmetric cost.

**The reward function**, stated plainly:

| Outcome | Reward |
| --- | --- |
| Fraud completes unchecked | −100 − 50×value |
| BLOCK on fraud | +40 + 20×value |
| BLOCK on a genuine caller | −25 − 15×value |
| CHALLENGE catching fraud | +30 |
| CHALLENGE on a genuine caller | −6 |
| ALERT catching fraud | +25 |
| ALERT on a genuine caller | −12 |
| Genuine call completes untouched | +25 |
| Each WAIT step | −0.25 × (1 + value) |

**Measured against a fixed-threshold baseline** (5,000 simulated calls):

| | Fixed threshold | RL policy |
| --- | --- | --- |
| Average reward | 17.74 | 25.13 |
| Fraud caught | 100% | 100% |
| False block rate | 0.7% | 0.9% |

It reaches the same fraud-catch rate with materially less unnecessary
friction. That is the entire claim.

**Two safety mechanisms, because a learned policy should not be trusted
blindly:**

- *Visit gating.* Any state seen fewer than 200 times during training is
  treated as unreliable, and the system falls back to threshold logic there.
- *Band floor.* The policy may escalate above what the calibrated bands
  imply, never below. It can add caution; it cannot remove it.

**What it does not claim.** The policy is trained in a simulator, not on
real fraud calls — there is no labelled live-call reward signal available.
Score distributions in the simulator are matched to the detector's measured
behaviour, including the finding that unfamiliar voices produce noisier,
less separable scores. Say this openly rather than implying the policy
learned from real attacks.

---

## Known limitations — state these, do not hide them

- **Live call interception is not implemented.** That requires telecom or
  VoIP-level integration. What exists is the detection and decision engine
  such an integration would sit on top of.
- **Unfamiliar voices are harder.** On held-out fresh recordings, clones of
  a voice well represented in training were caught near-perfectly, while
  clones of an unfamiliar speaker frequently landed in the uncertain band.
  The three-band design is the response to this, not a workaround for it.
- **Speaker-level generalisation was not measurable.** The original dataset
  carries no speaker identifiers, so train/test splits could not be grouped
  by speaker. Fresh self-recorded clips were used as the substitute test.
- **Leave-one-generator-out was not possible.** The fake folder is flat,
  with no per-synthesis-method subfolders.
- **The original dataset had a sample-rate and duration shortcut.** Detected
  at audit, and neutralised by resampling to 16 kHz, fixed-length chunking,
  and peak normalisation.

---

## Troubleshooting

**Model file not found** — `app.py` looks for `model_v2.pt`. Override with
`set VG_MODEL=model.pt` (Windows) before running.

**"Could not read that audio format"** — librosa needs ffmpeg for mp3/m4a.
Use WAV or FLAC, or install ffmpeg.

**Record button does nothing on the phone** — expected over plain http on a
LAN address. Use file upload.

**Port 5000 already in use** — change the port at the bottom of `app.py`.

**Phone cannot reach the laptop** — Windows Firewall usually blocks inbound
port 5000 on first run. Allow Python through the firewall for private
networks, or use a laptop hotspot.
