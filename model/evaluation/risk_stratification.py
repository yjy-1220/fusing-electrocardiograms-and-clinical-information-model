"""Three-tier risk stratification: dual-threshold search with Pareto-front filtering.

Implements the method described in the paper (Section 2.4):

  - Raw model probabilities p in [0, 1] define three tiers with a cutoff pair
    (t_low, t_high), 0 <= t_low < t_high <= 1:

        low-risk           p <= t_low          (rule-out group)
        intermediate-risk  t_low < p < t_high  (grey zone)
        high-risk          p >= t_high         (rule-in group)

  - Candidate cutoff pairs are generated on the internal-validation set and
    filtered by the predefined clinical constraints:
        rule-out purity  (NPV of the low-risk group)   >= rule_out_purity (0.98)
        rule-in purity   (PPV of the high-risk group)  >= rule_in_purity  (0.55)
        minimum population fraction in each extreme group               (0.04)
        combined extreme-group coverage (low + high)   >= min_coverage   (0.50)

  - Pareto-front filtering balances the clinical objectives (NPV_low,
    PPV_high, extreme-group coverage). If no pair satisfies every constraint,
    the weighted optimal solution balancing purity and coverage is selected
    from the entire Pareto front.

  - The selected cutoffs are rounded to clinically intuitive integer
    percentages (e.g. 5% / 75%) by default, which is how the paper obtains the
    final three-tier scheme: low-risk (<=5%), intermediate-risk (5%-75%),
    high-risk (>=75%).
"""
import numpy as np


def group_metrics(y_true, probs, t_low, t_high):
    """Evaluate the three-tier groups defined by the cutoff pair (t_low, t_high).

    Returns group sizes / fractions, OMI counts, per-group OMI prevalence,
    rule-out purity (NPV of the low-risk group), rule-in purity (PPV of the
    high-risk group), missed-diagnosis rate, rule-out sensitivity and the
    combined extreme-group coverage.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probs, dtype=float)
    n = int(len(y))
    low = p <= float(t_low)
    high = p >= float(t_high)
    mid = ~low & ~high
    n_low, n_mid, n_high = int(low.sum()), int(mid.sum()), int(high.sum())
    omi_low = int(y[low].sum())
    omi_mid = int(y[mid].sum())
    omi_high = int(y[high].sum())
    total_omi = int(y.sum())
    total_nomi = n - total_omi

    def _rate(num, den):
        return float(num / den) if den > 0 else float("nan")

    return {
        "t_low": float(t_low),
        "t_high": float(t_high),
        "n": n,
        "n_low": n_low, "n_mid": n_mid, "n_high": n_high,
        "frac_low": float(n_low / n) if n else float("nan"),
        "frac_mid": float(n_mid / n) if n else float("nan"),
        "frac_high": float(n_high / n) if n else float("nan"),
        "omi_low": omi_low, "omi_mid": omi_mid, "omi_high": omi_high,
        "nomi_low": n_low - omi_low, "nomi_mid": n_mid - omi_mid, "nomi_high": n_high - omi_high,
        "prevalence_low": _rate(omi_low, n_low),
        "prevalence_mid": _rate(omi_mid, n_mid),
        "prevalence_high": _rate(omi_high, n_high),
        # rule-out purity = NPV of the low-risk group = non-OMI / low-risk
        "npv_low": _rate(n_low - omi_low, n_low),
        # rule-in purity = PPV of the high-risk group = OMI / high-risk
        "ppv_high": _rate(omi_high, n_high),
        "missed_rate": _rate(omi_low, total_omi),
        "rule_out_sensitivity": _rate(total_omi - omi_low, total_omi),
        "coverage": float((n_low + n_high) / n) if n else float("nan"),
    }


def _threshold_grid(probs, max_candidates: int = 200):
    """Build the candidate cutoff grid from the empirical probability values."""
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    uniq = np.unique(p)
    if len(uniq) <= max_candidates:
        return uniq
    qs = np.linspace(0.0, 1.0, max_candidates)
    return np.unique(np.quantile(uniq, qs))


def _safe(obj_key, m, higher_is_better: bool):
    """NaN-safe objective value: missing / NaN collapses to -inf for maximisation."""
    v = m.get(obj_key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return -np.inf
    return float(v) if higher_is_better else -float(v)


def _pareto_filter(solutions, objectives):
    """Keep the non-dominated solutions (minimise -> pass higher_is_better=False).

    objectives: list of (metric_key, higher_is_better).
    """
    keep = []
    vals = [
        [_safe(k, s["metrics"], h) for k, h in objectives]
        for s in solutions
    ]
    for i, vi in enumerate(vals):
        dominated = False
        for j, vj in enumerate(vals):
            if i == j:
                continue
            d = np.asarray(vj, dtype=float) - np.asarray(vi, dtype=float)
            if np.all(d >= -1e-12) and np.any(d > 1e-12):
                dominated = True
                break
        if not dominated:
            keep.append(solutions[i])
    return keep


def search_dual_thresholds(
    y_true,
    probs,
    rule_out_purity: float = 0.98,
    rule_in_purity: float = 0.55,
    min_extreme_frac: float = 0.04,
    min_coverage: float = 0.50,
    purity_weight: float = 0.5,
    round_to_int: bool = True,
    max_candidates: int = 200,
    verbose: bool = True,
):
    """Search the three-tier cutoff pair (t_low, t_high) on the internal-validation set.

    Follows the paper's Section 2.4 workflow:
      1. enumerate candidate cutoff pairs on the empirical probability grid;
      2. keep pairs meeting the clinical constraints; if none does, fall back
         to the whole candidate pool;
      3. Pareto-filter the candidates over (NPV_low, PPV_high, coverage) and
         pick the weighted optimal solution (purity_weight * (NPV+PPV) +
         (1-purity_weight) * coverage);
      4. round the cutoffs to clinically intuitive integer percentages
         (e.g. 5% / 75%) and re-evaluate the groups.

    Returns a dict with the chosen cutoffs, the final group metrics, the
    applied constraints, and whether all constraints were simultaneously
    satisfiable before rounding.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probs, dtype=float)
    grid = _threshold_grid(p, max_candidates=max_candidates)

    constraints = {
        "rule_out_purity": float(rule_out_purity),
        "rule_in_purity": float(rule_in_purity),
        "min_extreme_frac": float(min_extreme_frac),
        "min_coverage": float(min_coverage),
        "purity_weight": float(purity_weight),
    }

    candidates = []
    feasible_any = False
    for i, t_low in enumerate(grid):
        for j in range(i + 1, len(grid)):
            t_high = grid[j]
            m = group_metrics(y, p, t_low, t_high)
            feasible = (
                m["npv_low"] >= rule_out_purity
                and m["ppv_high"] >= rule_in_purity
                and m["frac_low"] >= min_extreme_frac
                and m["frac_high"] >= min_extreme_frac
                and m["coverage"] >= min_coverage
            )
            candidates.append({"metrics": m, "feasible": bool(feasible)})
            feasible_any = feasible_any or bool(feasible)

    objectives = [("npv_low", True), ("ppv_high", True), ("coverage", True)]

    def _score(c):
        m = c["metrics"]
        s_npv = _safe("npv_low", m, True)
        s_ppv = _safe("ppv_high", m, True)
        s_cov = _safe("coverage", m, True)
        return purity_weight * (s_npv + s_ppv) + (1.0 - purity_weight) * s_cov

    if feasible_any:
        pool = [c for c in candidates if c["feasible"]]
        front = _pareto_filter(pool, objectives)
        chosen = max(front, key=_score)
    else:
        front = _pareto_filter(candidates, objectives)
        chosen = max(front, key=_score)

    t_low = float(chosen["metrics"]["t_low"])
    t_high = float(chosen["metrics"]["t_high"])
    if round_to_int:
        t_low = max(0.0, min(1.0, float(round(t_low * 100)) / 100.0))
        t_high = max(0.0, min(1.0, float(round(t_high * 100)) / 100.0))
        if t_low >= t_high:
            t_high = min(1.0, t_low + 0.01)
        final_metrics = group_metrics(y, p, t_low, t_high)
    else:
        final_metrics = dict(chosen["metrics"])
        t_low, t_high = final_metrics["t_low"], final_metrics["t_high"]

    if verbose:
        print("[RiskStrat] dual-threshold search over %d candidate pairs "
              "(satisfied-all=%s)" % (len(candidates), feasible_any))
        print("[RiskStrat] cutoffs: low-risk <= %.0f%%, high-risk >= %.0f%%"
              % (t_low * 100, t_high * 100))
        print("[RiskStrat] groups: low n=%d (%.2f%%, OMI=%d, NPV=%.3f) | "
              "mid n=%d (%.2f%%) | high n=%d (%.2f%%, OMI=%d, PPV=%.3f)"
              % (final_metrics["n_low"], final_metrics["frac_low"] * 100,
                 final_metrics["omi_low"], final_metrics["npv_low"],
                 final_metrics["n_mid"], final_metrics["frac_mid"] * 100,
                 final_metrics["n_high"], final_metrics["frac_high"] * 100,
                 final_metrics["omi_high"], final_metrics["ppv_high"]))
        print("[RiskStrat] coverage=%.2f%%, rule-out sensitivity=%.3f, "
              "missed rate=%.3f%%"
              % (final_metrics["coverage"] * 100,
                 final_metrics["rule_out_sensitivity"],
                 final_metrics["missed_rate"] * 100))

    return {
        "t_low": t_low,
        "t_high": t_high,
        "metrics": final_metrics,
        "constraints": constraints,
        "satisfied_all": bool(feasible_any),
    }


def evaluate_risk_tiers(y_true, probs, t_low: float = 0.05, t_high: float = 0.75):
    """Evaluate a fixed three-tier scheme (e.g. the final 5% / 75% cutoffs) on a dataset.

    Returns the same group metrics dict as ``group_metrics``.
    """
    return group_metrics(y_true, probs, float(t_low), float(t_high))
