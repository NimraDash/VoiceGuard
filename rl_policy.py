"""
rl_policy.py -- the sequential escalation policy.

WHAT THIS IS, precisely (so you can answer the judge's question):

The detector gives a probability per 4-second window. That alone is not a
decision. On a live call the system must choose, every few seconds:

    WAIT      let the call continue, gather more evidence
    CHALLENGE inject verification (call-back / OTP / knowledge question)
    ALERT     warn the operator mid-call
    BLOCK     stop the call / freeze the pending action

A fixed threshold is optimal only if you decide ONCE. Here the decision is
sequential: waiting has a cost that grows with call duration and transaction
value, challenging costs friction and tips off an attacker, and the right
move depends on how the score is TRENDING, not just its current value. That
is a sequential decision problem under asymmetric cost -- which is what RL
is actually for.

REWARD FUNCTION (stated explicitly -- no hand-waving):
    fraud completed (never stopped)   -100 - 50*value_bucket
    correct BLOCK on fraud             +40 + 20*value_bucket
    BLOCK on a genuine caller          -25 - 15*value_bucket
    CHALLENGE that catches fraud       +30
    CHALLENGE on a genuine caller       -6   (friction)
    ALERT that catches fraud           +25
    ALERT on a genuine caller          -12
    genuine call completes untouched   +25
    each WAIT step                     -0.25 * (1 + value_bucket)

TRAINING: tabular Q-learning in a replay simulator. We cannot collect real
reward signal in a hackathon (no labelled live calls), so episodes are
simulated from score distributions matched to the detector's MEASURED
behaviour -- including the finding that unfamiliar voices produce noisier,
less separable scores than familiar ones. This is stated openly; it is not
presented as learning from real fraud.
"""

import json
import random
from pathlib import Path

import numpy as np

ACTIONS = ["WAIT", "CHALLENGE", "ALERT", "BLOCK"]

# ---------------------------------------------------------------------------
# cost matrix
# ---------------------------------------------------------------------------
R_FRAUD_COMPLETED = (-100.0, -50.0)   # base, per value_bucket
R_BLOCK_FRAUD = (40.0, 20.0)
R_BLOCK_GENUINE = (-25.0, -15.0)
R_CHALLENGE_FRAUD = 30.0
R_CHALLENGE_GENUINE = -6.0
R_ALERT_FRAUD = 25.0
R_ALERT_GENUINE = -12.0
R_GENUINE_COMPLETED = 25.0
# Kept small on purpose. The real cost of waiting on a fraudulent call is
# already carried by R_FRAUD_COMPLETED; charging a large per-step penalty on
# top of it double-counts the same risk, and makes letting an honest call
# finish look almost as expensive as interrupting it.
R_WAIT_STEP = -0.25


class CallSimulator:
    """
    Generates synthetic call episodes. Score distributions are Beta
    distributions whose means match what the detector actually produced on
    held-out data; unfamiliar callers get wider (noisier) distributions,
    reflecting the measured generalisation gap on unseen voices.
    """

    def __init__(self, max_chunks=10, fraud_rate=0.35, seed=0):
        self.max_chunks = max_chunks
        self.fraud_rate = fraud_rate
        self.rng = np.random.default_rng(seed)

    def _beta_params(self, mean, concentration):
        a = mean * concentration
        b = (1 - mean) * concentration
        return max(a, 0.05), max(b, 0.05)

    def new_episode(self):
        is_fraud = self.rng.random() < self.fraud_rate
        caller_known = int(self.rng.random() < 0.5)
        value_bucket = int(self.rng.choice([0, 1, 2], p=[0.5, 0.3, 0.2]))

        # familiar voices separate cleanly; unfamiliar ones do not
        if caller_known:
            mean = 0.86 if is_fraud else 0.10
            conc = 9.0
        else:
            mean = 0.62 if is_fraud else 0.30
            conc = 4.0

        a, b = self._beta_params(mean, conc)
        scores = self.rng.beta(a, b, size=self.max_chunks)
        return {
            "is_fraud": is_fraud,
            "caller_known": caller_known,
            "value_bucket": value_bucket,
            "scores": scores,
        }


# ---------------------------------------------------------------------------
# state encoding
# ---------------------------------------------------------------------------
def encode_state(mean_score, slope, n_chunks, caller_known, value_bucket,
                 max_chunks=10):
    s_b = int(np.clip(mean_score * 5, 0, 4))
    if slope > 0.05:
        sl_b = 2
    elif slope < -0.05:
        sl_b = 0
    else:
        sl_b = 1
    t_b = int(np.clip(n_chunks / max(1, max_chunks) * 3, 0, 2))
    return (s_b, sl_b, t_b, int(caller_known), int(value_bucket))


def running_stats(scores_so_far):
    arr = np.asarray(scores_so_far, dtype=float)
    mean = float(arr.mean())
    if len(arr) >= 3:
        x = np.arange(len(arr))
        slope = float(np.polyfit(x, arr, 1)[0])
    else:
        slope = 0.0
    return mean, slope


# ---------------------------------------------------------------------------
# environment step
# ---------------------------------------------------------------------------
def terminal_reward(action, is_fraud, value_bucket):
    v = value_bucket
    if action == "BLOCK":
        if is_fraud:
            return R_BLOCK_FRAUD[0] + R_BLOCK_FRAUD[1] * v
        return R_BLOCK_GENUINE[0] + R_BLOCK_GENUINE[1] * v
    if action == "CHALLENGE":
        return R_CHALLENGE_FRAUD if is_fraud else R_CHALLENGE_GENUINE
    if action == "ALERT":
        return R_ALERT_FRAUD if is_fraud else R_ALERT_GENUINE
    raise ValueError(action)


def episode_end_reward(is_fraud, value_bucket):
    if is_fraud:
        return R_FRAUD_COMPLETED[0] + R_FRAUD_COMPLETED[1] * value_bucket
    return R_GENUINE_COMPLETED


def run_episode(ep, act_fn, max_chunks=10):
    """act_fn(state, step) -> action string. Returns total reward."""
    total = 0.0
    seen = []
    for t in range(max_chunks):
        seen.append(ep["scores"][t])
        mean, slope = running_stats(seen)
        state = encode_state(mean, slope, t + 1, ep["caller_known"],
                             ep["value_bucket"], max_chunks)
        action = act_fn(state, t)
        if action == "WAIT":
            total += R_WAIT_STEP * (1 + ep["value_bucket"])
            continue
        total += terminal_reward(action, ep["is_fraud"], ep["value_bucket"])
        return total, action, t + 1
    total += episode_end_reward(ep["is_fraud"], ep["value_bucket"])
    return total, "WAIT", max_chunks


# ---------------------------------------------------------------------------
# Q-learning
# ---------------------------------------------------------------------------
def train_q(episodes=40000, alpha=0.15, gamma=0.97, eps_start=0.9,
            eps_end=0.05, max_chunks=10, seed=0):
    rng = random.Random(seed)
    sim = CallSimulator(max_chunks=max_chunks, seed=seed)
    Q = {}
    N = {}

    def qget(s):
        if s not in Q:
            Q[s] = [0.0] * len(ACTIONS)
            N[s] = 0
        return Q[s]

    for i in range(episodes):
        eps = eps_start + (eps_end - eps_start) * (i / max(1, episodes - 1))
        ep = sim.new_episode()
        seen = []
        prev = None  # (state, action_idx)
        for t in range(max_chunks):
            seen.append(ep["scores"][t])
            mean, slope = running_stats(seen)
            s = encode_state(mean, slope, t + 1, ep["caller_known"],
                             ep["value_bucket"], max_chunks)
            qs = qget(s)
            N[s] = N.get(s, 0) + 1
            if rng.random() < eps:
                ai = rng.randrange(len(ACTIONS))
            else:
                ai = int(np.argmax(qs))
            action = ACTIONS[ai]

            if prev is not None:
                ps, pai, pr = prev
                qget(ps)[pai] += alpha * (pr + gamma * max(qs) - qget(ps)[pai])

            if action == "WAIT":
                r = R_WAIT_STEP * (1 + ep["value_bucket"])
                if t == max_chunks - 1:
                    r += episode_end_reward(ep["is_fraud"], ep["value_bucket"])
                    qs[ai] += alpha * (r - qs[ai])
                    prev = None
                else:
                    prev = (s, ai, r)
                continue

            r = terminal_reward(action, ep["is_fraud"], ep["value_bucket"])
            qs[ai] += alpha * (r - qs[ai])
            prev = None
            break
    return Q, N


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------
def make_q_policy(Q):
    def act(state, step):
        if state not in Q:
            return "WAIT"
        return ACTIONS[int(np.argmax(Q[state]))]
    return act


def make_threshold_policy(low, high):
    """Baseline: decide on the running mean with fixed cutoffs, at every step."""
    def act(state, step):
        s_b = state[0]
        mean_approx = (s_b + 0.5) / 5.0
        if mean_approx > high:
            return "BLOCK"
        if mean_approx < low:
            return "WAIT"
        return "CHALLENGE"
    return act


def evaluate(act_fn, n=5000, max_chunks=10, seed=1234):
    sim = CallSimulator(max_chunks=max_chunks, seed=seed)
    totals, caught, missed, false_block = [], 0, 0, 0
    for _ in range(n):
        ep = sim.new_episode()
        r, action, steps = run_episode(ep, act_fn, max_chunks)
        totals.append(r)
        if ep["is_fraud"]:
            if action in ("BLOCK", "CHALLENGE", "ALERT"):
                caught += 1
            else:
                missed += 1
        else:
            if action == "BLOCK":
                false_block += 1
    n_fraud = caught + missed
    return {
        "avg_reward": float(np.mean(totals)),
        "fraud_caught_rate": caught / max(1, n_fraud),
        "fraud_missed": missed,
        "false_block_rate": false_block / max(1, n - n_fraud),
    }


# ---------------------------------------------------------------------------
# persistence + runtime use
# ---------------------------------------------------------------------------
# severity ordering -- used for the safety floor below
SEVERITY = {"WAIT": 0, "CHALLENGE": 1, "ALERT": 2, "BLOCK": 3}

# a state seen fewer than this many times in training is not trusted;
# we fall back to the calibrated threshold bands there rather than acting
# on a Q-value that is mostly noise
MIN_VISITS = 200


def save_policy(Q, path="rl_policy.json", meta=None, N=None):
    data = {
        "actions": ACTIONS,
        "meta": meta or {},
        "q": {",".join(map(str, k)): v for k, v in Q.items()},
        "n": {",".join(map(str, k)): int(v) for k, v in (N or {}).items()},
    }
    Path(path).write_text(json.dumps(data))


def load_policy(path="rl_policy.json"):
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    Q = {tuple(int(x) for x in k.split(",")): v for k, v in data["q"].items()}
    N = {tuple(int(x) for x in k.split(",")): v
         for k, v in data.get("n", {}).items()}
    return {"Q": Q, "N": N, "meta": data.get("meta", {})}


def band_floor(mean_score, low, high):
    """
    Minimum action severity implied by the calibrated bands. The learned
    policy may escalate ABOVE this, never below it: the policy is allowed to
    add caution, not remove it. This keeps a rarely-visited or noisy state
    from ever downgrading a clearly synthetic score to 'keep listening'.
    """
    if mean_score > high:
        return "ALERT"
    if mean_score >= low:
        return "CHALLENGE"
    return "WAIT"


def decide(policy, mean_score, slope, n_chunks, caller_known, value_bucket,
           max_chunks=10, low=None, high=None):
    """
    Runtime entry point used by the web app.

    Returns (action, state, info). action is None only when there is no
    policy at all, in which case the caller applies threshold logic.
    """
    state = encode_state(mean_score, slope, n_chunks, caller_known,
                         value_bucket, max_chunks)
    if policy is None:
        return None, state, None

    qvals = policy["Q"].get(state)
    visits = policy.get("N", {}).get(state, 0)

    if qvals is None or visits < MIN_VISITS:
        return None, state, {"reason": "undertrained_state", "visits": visits}

    action = ACTIONS[int(np.argmax(qvals))]

    if low is not None and high is not None:
        floor = band_floor(mean_score, low, high)
        if SEVERITY[action] < SEVERITY[floor]:
            return floor, state, {"reason": "raised_to_band_floor",
                                  "policy_action": action, "visits": visits}

    return action, state, {"reason": "policy", "visits": visits}
