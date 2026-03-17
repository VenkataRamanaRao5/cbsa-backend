"""
tests/evaluation.py — Quantitative Validation Engine

Computes security-relevant metrics for the CBSA continuous authentication system:

    Metric       Definition
    --------     ----------------------------------------------------------
    Accuracy     (TP + TN) / (TP + TN + FP + FN)
    FAR          False Accept Rate  = FP / (FP + TN)
                 Fraction of ATTACKER events classified as ACCEPT
    FRR          False Reject Rate  = FN / (FN + TP)
                 Fraction of GENUINE events classified as REJECT
    EER          Equal Error Rate   = FAR at the threshold where FAR ≈ FRR
                 Lower is better; EER ≤ 10% is typical target for biometrics

Decision rule:
    ACCEPT  if trust_score > ACCEPT_THRESHOLD (default: 0.65)
    REJECT  otherwise

Intended use:
    from tests.evaluation import evaluate, EvaluationResult
    result = evaluate(genuine_results, attack_results)
    result.print_report()

Input:
    Each result list is a list of dicts produced by tests/runner.run_pipeline():
        {
            "event":      int,
            "similarity": float,
            "short_drift": float,
            "stability":  float,
            "anomaly":    float,
            "trust":      float,   ← used for classification
            "decision":   str,
            "proto_id":   int | None,
            "n_protos":   int,
        }

Warmup exclusion:
    The first WARMUP_SKIP_EVENTS events from genuine scenarios are excluded
    from FAR/FRR computation because trust has not had time to stabilise after
    quarantine promotion. Including warmup events in FRR would unfairly inflate
    the rejection rate for a system that is designed to withhold judgment during
    enrollment.

EER computation:
    We scan thresholds from 0 to 1 in steps of 0.001 and find the value t*
    that minimises |FAR(t*) - FRR(t*)|.  The EER is defined as:
        EER = (FAR(t*) + FRR(t*)) / 2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Number of warm-up events to skip from genuine results when computing FRR
WARMUP_SKIP_EVENTS: int = 20

# Default decision threshold
ACCEPT_THRESHOLD: float = 0.65


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """Complete quantitative evaluation of the CBSA system."""

    # Counts
    n_genuine_events:   int = 0   # events from genuine user (after warmup)
    n_attack_events:    int = 0   # events from attacker
    true_positives:     int = 0   # genuine → ACCEPT  ✓
    false_negatives:    int = 0   # genuine → REJECT  ✗  (FRR numerator)
    true_negatives:     int = 0   # attack  → REJECT  ✓
    false_positives:    int = 0   # attack  → ACCEPT  ✗  (FAR numerator)

    # Derived metrics (computed in evaluate())
    accuracy: float = 0.0
    far:      float = 0.0   # False Accept Rate
    frr:      float = 0.0   # False Reject Rate
    eer:      float = 0.0   # Equal Error Rate

    # Trust score distributions (for validation report)
    avg_trust_genuine: float = 0.0
    avg_trust_attack:  float = 0.0
    gat_usage_rate:    float = 0.0

    # Threshold used for accept/reject decision
    threshold: float = ACCEPT_THRESHOLD

    def print_report(self, title: str = "SYSTEM VALIDATION REPORT") -> None:
        """Print formatted system validation report."""
        width = 60
        line = "=" * width
        print(f"\n{line}")
        print(f" {title}")
        print(line)
        print(f"  Genuine events evaluated : {self.n_genuine_events}")
        print(f"  Attack  events evaluated : {self.n_attack_events}")
        print(f"  Decision threshold       : trust > {self.threshold:.2f} -> ACCEPT")
        print()
        print(f"  True  Positives  (TP)    : {self.true_positives}")
        print(f"  False Negatives  (FN)    : {self.false_negatives}")
        print(f"  True  Negatives  (TN)    : {self.true_negatives}")
        print(f"  False Positives  (FP)    : {self.false_positives}")
        print()
        print(f"  Accuracy                 : {self.accuracy * 100:.2f}%")
        print(f"  FAR  (False Accept Rate) : {self.far  * 100:.2f}%")
        print(f"  FRR  (False Reject Rate) : {self.frr  * 100:.2f}%")
        print(f"  EER  (Equal Error Rate)  : {self.eer  * 100:.2f}%")
        print()
        print(f"  Avg trust (genuine)      : {self.avg_trust_genuine:.4f}")
        print(f"  Avg trust (attack)       : {self.avg_trust_attack:.4f}")
        print(f"  GAT usage rate           : {self.gat_usage_rate * 100:.1f}%")
        print()
        status = _system_status(self)
        print(f"  Status                   : {status}")
        print(line)


# ── Core evaluation function ──────────────────────────────────────────────────

def evaluate(
    genuine_results:  List[dict],
    attack_results:   List[dict],
    threshold:        float = ACCEPT_THRESHOLD,
    warmup_skip:      int   = WARMUP_SKIP_EVENTS,
) -> EvaluationResult:
    """
    Compute FAR, FRR, EER, and accuracy from scenario results.

    Parameters
    ----------
    genuine_results : Per-event dicts from scenario_genuine (or scenario_drift
                      stable phase). Ground truth: every event should be ACCEPTED.
    attack_results  : Per-event dicts from scenario_attack.
                      Ground truth: every event should be REJECTED.
    threshold       : trust_score above which a decision is ACCEPT.
    warmup_skip     : Skip first N genuine events (quarantine enrollment phase).

    Returns
    -------
    EvaluationResult with all metrics populated.
    """
    result = EvaluationResult(threshold=threshold)

    # ── Genuine user evaluation ───────────────────────────────────────────────
    genuine_eval = genuine_results[warmup_skip:]   # skip quarantine warmup
    result.n_genuine_events = len(genuine_eval)

    genuine_trusts = [r["trust"] for r in genuine_eval]
    result.avg_trust_genuine = float(np.mean(genuine_trusts)) if genuine_trusts else 0.0

    for r in genuine_eval:
        if r["trust"] > threshold:
            result.true_positives += 1    # correctly accepted
        else:
            result.false_negatives += 1   # incorrectly rejected

    # ── Attack evaluation ─────────────────────────────────────────────────────
    result.n_attack_events = len(attack_results)

    attack_trusts = [r["trust"] for r in attack_results]
    result.avg_trust_attack = float(np.mean(attack_trusts)) if attack_trusts else 1.0

    for r in attack_results:
        if r["trust"] > threshold:
            result.false_positives += 1   # incorrectly accepted (security breach)
        else:
            result.true_negatives += 1    # correctly rejected

    # ── Derived metrics ───────────────────────────────────────────────────────
    total = result.true_positives + result.false_negatives + \
            result.true_negatives + result.false_positives

    result.accuracy = (result.true_positives + result.true_negatives) / total \
        if total > 0 else 0.0

    result.far = result.false_positives / result.n_attack_events \
        if result.n_attack_events > 0 else 0.0

    result.frr = result.false_negatives / result.n_genuine_events \
        if result.n_genuine_events > 0 else 0.0

    result.eer = _compute_eer(genuine_trusts, attack_trusts)

    return result


def _compute_eer(
    genuine_trusts: List[float],
    attack_trusts:  List[float],
) -> float:
    """
    Find the Equal Error Rate (EER) by scanning thresholds in [0, 1].

    EER = value of FAR (≈ FRR) at the threshold where |FAR - FRR| is minimised.

    Returns 0.0 if either list is empty.
    """
    if not genuine_trusts or not attack_trusts:
        return 0.0

    g = np.asarray(genuine_trusts, dtype=float)
    a = np.asarray(attack_trusts, dtype=float)

    thresholds = np.linspace(0.0, 1.0, 1001)
    best_eer  = 1.0
    best_diff = float("inf")

    for t in thresholds:
        far = float(np.mean(a > t))   # fraction of attack events accepted
        frr = float(np.mean(g <= t))  # fraction of genuine events rejected
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_eer  = (far + frr) / 2.0

    return float(best_eer)


def _system_status(result: EvaluationResult) -> str:
    """
    Return a coarse status string based on EER and trust separation.

    Thresholds are research-grade baselines for touch-based biometrics:
      EXCELLENT : EER < 5%  and trust separation > 0.25
      GOOD      : EER < 10% and trust separation > 0.15
      NEEDS TUNING : otherwise
    """
    separation = result.avg_trust_genuine - result.avg_trust_attack
    if result.eer < 0.05 and separation > 0.25:
        return "EXCELLENT"
    if result.eer < 0.10 and separation > 0.15:
        return "GOOD"
    if result.eer < 0.15:
        return "STABLE"
    return "NEEDS TUNING"


# ── Scenario-level helpers ────────────────────────────────────────────────────

def evaluate_scenario_pair(
    genuine_results: List[dict],
    attack_results:  List[dict],
    threshold:       float = ACCEPT_THRESHOLD,
    title:           str   = "SYSTEM VALIDATION REPORT",
    print_report:    bool  = True,
) -> EvaluationResult:
    """Convenience wrapper: evaluate and optionally print."""
    result = evaluate(genuine_results, attack_results, threshold=threshold)
    if print_report:
        result.print_report(title)
    return result


def evaluate_threshold_sweep(
    genuine_results: List[dict],
    attack_results:  List[dict],
    n_steps:         int = 20,
) -> List[dict]:
    """
    Evaluate at multiple thresholds; useful for ROC curve data.

    Returns list of dicts: [{threshold, far, frr, accuracy}, ...]
    """
    rows = []
    for t in np.linspace(0.0, 1.0, n_steps + 1):
        r = evaluate(genuine_results, attack_results, threshold=float(t))
        rows.append({
            "threshold": round(float(t), 3),
            "far":       round(r.far,  4),
            "frr":       round(r.frr,  4),
            "accuracy":  round(r.accuracy, 4),
        })
    return rows
