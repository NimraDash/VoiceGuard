"""
risk_bands.py -- ONE place that turns a raw model probability into a
decision. Import this everywhere instead of writing `if p > 0.5` again.

Thresholds below were picked by looking at your actual validation scores
(training run: val EER 0.102, thr 0.772) and your fresh out-of-distribution
test (13 real / 27 fake, separation +0.356). They are a starting point, not
a law -- see `suggest_thresholds()` at the bottom to recompute them from any
labeled set you have.
"""

from dataclasses import dataclass

# --- current working thresholds ---------------------------------------------
LOW = 0.5018    # calibrated by 09_calibrate.py
HIGH = 0.6112   # calibrated by 09_calibrate.py
# between LOW and HIGH       -> UNCERTAIN


@dataclass
class RiskVerdict:
    probability: float
    band: str          # "GENUINE" | "UNCERTAIN" | "SYNTHETIC"
    message: str
    action: str         # what the app should do


def classify(p_fake: float, low: float = LOW, high: float = HIGH) -> RiskVerdict:
    if p_fake < low:
        return RiskVerdict(
            probability=p_fake,
            band="GENUINE",
            message="Voice appears genuine.",
            action="ALLOW",
        )
    if p_fake > high:
        return RiskVerdict(
            probability=p_fake,
            band="SYNTHETIC",
            message="High probability of AI-generated / cloned voice. "
                    "Do not act on financial or confidential instructions "
                    "from this call without independent verification.",
            action="BLOCK_AND_ALERT",
        )
    return RiskVerdict(
        probability=p_fake,
        band="UNCERTAIN",
        message="Cannot confidently confirm this voice is genuine. "
                "Recommend a call-back on a known number before proceeding.",
        action="CHALLENGE",
    )


def suggest_thresholds(real_scores, fake_scores, target_false_alarm=0.03):
    """
    Optional helper: recompute LOW/HIGH from any labeled score set.
    - LOW is picked so that at most `target_false_alarm` fraction of REAL
      clips score above it (keep false accusations rare).
    - HIGH is picked so that at most `target_false_alarm` fraction of FAKE
      clips score below it (keep missed fakes rare).
    Everything between the two is "uncertain" by construction.
    """
    import numpy as np
    real_scores = np.sort(np.asarray(real_scores))
    fake_scores = np.sort(np.asarray(fake_scores))[::-1]

    low_idx = int(len(real_scores) * (1 - target_false_alarm))
    low_idx = min(max(low_idx, 0), len(real_scores) - 1)
    low = float(real_scores[low_idx])

    high_idx = int(len(fake_scores) * (1 - target_false_alarm))
    high_idx = min(max(high_idx, 0), len(fake_scores) - 1)
    high = float(fake_scores[high_idx])

    if high < low:
        # bands would overlap/invert on this small a sample; fall back to
        # the midpoint split so GENUINE/SYNTHETIC don't cross
        mid = (low + high) / 2
        low = high = mid
    return low, high


if __name__ == "__main__":
    # quick manual check
    for p in [0.0, 0.09, 0.24, 0.29, 0.30, 0.5, 0.56, 0.69, 0.75, 0.9, 1.0]:
        v = classify(p)
        print(f"p={p:.2f}  -> {v.band:<10} action={v.action}")
