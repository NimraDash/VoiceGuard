"""
10_train_rl.py -- train the escalation policy and prove it beats a fixed
threshold, or tell you it doesn't.

Run this ONCE. It writes rl_policy.json, which the web app loads.

    python 10_train_rl.py

If the policy does NOT beat the fixed-threshold baseline, this script says
so plainly. In that case, do not present it -- fall back to thresholds and
say so. A component you cannot show a win for is worse than no component.
"""

import argparse
import time

from rl_policy import (train_q, make_q_policy, make_threshold_policy,
                       evaluate, save_policy)
from risk_bands import LOW, HIGH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40000)
    ap.add_argument("--eval-n", type=int, default=5000)
    ap.add_argument("--out", default="rl_policy.json")
    args = ap.parse_args()

    print(f"training escalation policy ({args.episodes} simulated calls)...")
    t0 = time.time()
    Q, N = train_q(episodes=args.episodes)
    well_trained = sum(1 for s, c in N.items() if c >= 200)
    print(f"done in {time.time()-t0:.1f}s   states visited: {len(Q)}   "
          f"well-trained (>=200 visits): {well_trained}\n")

    q_pol = make_q_policy(Q)
    base_pol = make_threshold_policy(LOW, HIGH)

    q_res = evaluate(q_pol, n=args.eval_n)
    b_res = evaluate(base_pol, n=args.eval_n)

    print(f"{'metric':<24}{'fixed threshold':>18}{'RL policy':>14}")
    print("-" * 56)
    for k in ["avg_reward", "fraud_caught_rate", "false_block_rate"]:
        print(f"{k:<24}{b_res[k]:>18.3f}{q_res[k]:>14.3f}")
    print(f"{'fraud missed (count)':<24}{b_res['fraud_missed']:>18}"
          f"{q_res['fraud_missed']:>14}")
    print("-" * 56)

    delta = q_res["avg_reward"] - b_res["avg_reward"]
    print(f"\nreward improvement: {delta:+.2f} per call")

    if delta > 0:
        print("\n==> RL policy beats the fixed-threshold baseline on the "
              "stated cost matrix. Safe to present, WITH the caveat that "
              "training is simulator-based.")
        save_policy(Q, args.out, meta={
            "episodes": args.episodes,
            "baseline": b_res,
            "rl": q_res,
            "well_trained_states": well_trained,
            "thresholds_used_for_baseline": {"low": LOW, "high": HIGH},
        }, N=N)
        print(f"saved -> {args.out}")
    else:
        print("\n==> RL policy does NOT beat the baseline. Do not present it. "
              "The app will fall back to threshold logic automatically.")


if __name__ == "__main__":
    main()
