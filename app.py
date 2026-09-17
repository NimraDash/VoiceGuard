"""
app.py -- VoiceGuard web application.

Run:
    python app.py
Then open http://localhost:5000  (or http://<your-ip>:5000 from a phone on
the same wifi network).

Pipeline per request:
    audio -> 4s sliding windows (2s hop)
          -> 3-channel spectro features (mel / group-delay / delta-mel)
          -> EfficientNet-B0 -> per-window P(synthetic)
          -> temporal aggregation (mean + trend)
          -> risk band (genuine / uncertain / synthetic)
          -> RL escalation policy -> recommended action

Privacy: audio is scored in memory and never written to disk.
"""

import io
import os
import traceback
from pathlib import Path

import numpy as np
import torch
import librosa
from flask import Flask, jsonify, render_template, request

import importlib.util

BASE = Path(__file__).parent

spec = importlib.util.spec_from_file_location("train03", BASE / "03_train.py")
train03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train03)

from risk_bands import classify, LOW, HIGH
import rl_policy as rlp

MODEL_PATH = os.environ.get("VG_MODEL", str(BASE / "model_v2.pt"))
RL_PATH = str(BASE / "rl_policy.json")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_policy = None
_model_meta = {}


def get_model():
    global _model, _model_meta
    if _model is None:
        m = train03.build_model(pretrained=False).to(_device)
        ck = torch.load(MODEL_PATH, map_location=_device)
        m.load_state_dict(ck["model"])
        m.eval()
        _model = m
        _model_meta = ck.get("val", {})
        print(f"[voiceguard] model loaded from {MODEL_PATH} on {_device}")
    return _model


def get_policy():
    global _policy
    if _policy is None:
        _policy = rlp.load_policy(RL_PATH)
        if _policy:
            print(f"[voiceguard] escalation policy loaded "
                  f"({len(_policy['Q'])} states)")
        else:
            print("[voiceguard] no rl_policy.json -- using threshold logic")
    return _policy


def score_windows(y):
    """Return list of (start_seconds, probability) over 4s windows, 2s hop."""
    model = get_model()
    CL = train03.CHUNK_LEN
    SR = train03.SR
    if y.size < CL:
        y = np.pad(y, (0, CL - y.size))
    step = SR * 2
    out = []
    for start in range(0, max(1, len(y) - CL + 1), step):
        chunk = y[start:start + CL]
        if len(chunk) < CL:
            chunk = np.pad(chunk, (0, CL - len(chunk)))
        peak = np.max(np.abs(chunk)) + 1e-9
        chunk = (chunk / peak).astype(np.float32)
        x = train03.featurise(chunk)
        xt = torch.from_numpy(x).unsqueeze(0).to(_device)
        with torch.no_grad():
            p = torch.softmax(model(xt), 1)[0, 1].item()
        out.append((start / SR, float(p)))
    return out


def trend(values):
    if len(values) < 3:
        return 0.0
    x = np.arange(len(values))
    return float(np.polyfit(x, np.asarray(values), 1)[0])


ACTION_COPY = {
    "WAIT": ("No action needed yet",
             "Nothing in this audio suggests a synthetic voice so far. The "
             "system keeps scoring while the call continues."),
    "CHALLENGE": ("Verify who you are speaking to",
                  "Confirm identity on a number you already have, or use a "
                  "second factor, before acting on anything said on this "
                  "call."),
    "ALERT": ("Treat this call as suspicious",
              "This voice shows signs of being AI-generated. Do not act on "
              "instructions from this call, and tell your supervisor before "
              "approving anything."),
    "BLOCK": ("This voice is probably AI-generated",
              "Strong signs of synthetic speech. Consider ending the call "
              "and calling the person back on a number you already trust. "
              "Do not approve any transfer or share confidential details."),
}


@app.route("/")
def index():
    return render_template("index.html", low=LOW, high=HIGH)


@app.route("/api/health")
def health():
    get_model()
    pol = get_policy()
    return jsonify({
        "model": Path(MODEL_PATH).name,
        "device": _device,
        "val_metrics": _model_meta,
        "thresholds": {"low": LOW, "high": HIGH},
        "policy_loaded": pol is not None,
        "policy_meta": (pol or {}).get("meta", {}),
    })


@app.route("/api/score_window", methods=["POST"])
def score_window():
    """
    Lightweight endpoint used while recording is in progress. Scores ONE
    window and returns just the probability, so the interface can update
    live without re-running the whole file.
    """
    try:
        if "audio" not in request.files:
            return jsonify({"error": "no audio"}), 400
        raw = request.files["audio"].read()
        if not raw:
            return jsonify({"error": "empty"}), 400

        y, _ = librosa.load(io.BytesIO(raw), sr=train03.SR, mono=True)
        if y.size < train03.SR * 0.5:
            return jsonify({"error": "too short"}), 400

        # score only the most recent 4 seconds
        CL = train03.CHUNK_LEN
        if y.size < CL:
            y = np.pad(y, (0, CL - y.size))
        chunk = y[-CL:]
        peak = np.max(np.abs(chunk)) + 1e-9
        chunk = (chunk / peak).astype(np.float32)

        model = get_model()
        x = train03.featurise(chunk)
        xt = torch.from_numpy(x).unsqueeze(0).to(_device)
        with torch.no_grad():
            p = float(torch.softmax(model(xt), 1)[0, 1].item())

        return jsonify({"p": p, "band": classify(p).band,
                        "thresholds": {"low": LOW, "high": HIGH}})
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "scoring failed"}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio received. Choose a file or "
                                     "record a clip."}), 400
        raw = request.files["audio"].read()
        if not raw:
            return jsonify({"error": "That file is empty."}), 400

        caller_known = int(request.form.get("caller_known", 0))
        value_bucket = int(request.form.get("value_bucket", 0))

        try:
            y, _ = librosa.load(io.BytesIO(raw), sr=train03.SR, mono=True)
        except Exception:
            return jsonify({
                "error": "Could not read that audio format. Use a WAV or "
                         "FLAC file, or record with the button above."
            }), 400

        duration = len(y) / train03.SR
        if duration < 0.5:
            return jsonify({"error": "Clip is too short. Record at least "
                                     "one second."}), 400

        windows = score_windows(y)
        probs = [p for _, p in windows]
        mean_p = float(np.mean(probs))
        slope = trend(probs)
        verdict = classify(mean_p)

        policy = get_policy()
        action, state, info = rlp.decide(
            policy, mean_p, slope, len(probs), caller_known, value_bucket,
            low=LOW, high=HIGH)

        source = (info or {}).get("reason", "policy")
        if action is None:
            source = "threshold"
            action = {"GENUINE": "WAIT", "UNCERTAIN": "CHALLENGE",
                      "SYNTHETIC": "BLOCK"}[verdict.band]

        title, detail = ACTION_COPY[action]

        return jsonify({
            "probability": mean_p,
            "band": verdict.band,
            "message": verdict.message,
            "duration": duration,
            "trend": slope,
            "windows": [{"t": round(t, 2), "p": round(p, 4)}
                        for t, p in windows],
            "action": action,
            "action_title": title,
            "action_detail": detail,
            "action_source": source,
            "thresholds": {"low": LOW, "high": HIGH},
        })
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Analysis failed. Check the server "
                                 "console for details."}), 500


if __name__ == "__main__":
    import argparse
    import socket

    ap = argparse.ArgumentParser()
    ap.add_argument("--https", action="store_true",
                    help="serve over https with a self-signed certificate. "
                         "Needed for microphone access from a phone, since "
                         "browsers block getUserMedia on plain http.")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    get_model()
    get_policy()

    def lan_ip():
        """
        Find the address other devices on the network can reach. Opening a
        UDP socket toward an external address makes the OS pick the right
        outbound interface; nothing is actually sent.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = None
        finally:
            s.close()
        if not ip or ip.startswith("127."):
            try:
                for cand in socket.gethostbyname_ex(socket.gethostname())[2]:
                    if not cand.startswith("127."):
                        return cand
            except Exception:
                pass
            return "<run ipconfig to find your IP>"
        return ip

    ip = lan_ip()

    ssl_ctx = None
    scheme = "http"
    if args.https:
        try:
            import ssl as _ssl  # noqa: F401
            from cryptography import x509  # noqa: F401
            ssl_ctx = "adhoc"
            scheme = "https"
        except Exception:
            print("\n  --https needs the 'cryptography' package:")
            print("      pip install cryptography\n")
            raise SystemExit(1)

    print("\n  VoiceGuard running.")
    print(f"  On this machine:  {scheme}://localhost:{args.port}")
    print(f"  From your phone:  {scheme}://{ip}:{args.port}")
    if args.https:
        print("\n  The phone will warn about the certificate. Tap Advanced,")
        print("  then Proceed. This is expected for a self-signed cert.")
    else:
        print("\n  Microphone recording works on localhost only over http.")
        print("  For microphone access from a phone, restart with:")
        print("      python app.py --https")
    print()

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True,
            ssl_context=ssl_ctx)
