import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimizer import (
    generate_efficient_frontier,
    maximize_sharpe_ratio,
    minimize_variance,
    portfolio_metrics,
)

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "outputs" / "reports"


def load_inputs():
    mean = pd.read_csv(
        ROOT / "data" / "processed" / "annualised_mean_returns.csv",
        index_col=0,
    ).iloc[:, 0]
    cov = pd.read_csv(
        ROOT / "data" / "processed" / "annualised_covariance.csv",
        index_col=0,
    )
    cov = cov.loc[mean.index, mean.index]
    return mean, cov


def check_weights(weights, tolerance=1e-6):
    weights = np.asarray(weights, dtype=float)
    return {
        "finite": bool(np.all(np.isfinite(weights))),
        "sum": float(weights.sum()),
        "sum_ok": bool(abs(weights.sum() - 1.0) <= tolerance),
        "min_weight": float(weights.min()),
        "max_weight": float(weights.max()),
        "bounds_ok": bool(weights.min() >= -tolerance and weights.max() <= 1.0 + tolerance),
    }


def make_covariance(cov, scale):
    stressed = cov.copy()
    stressed.values[np.diag_indices_from(stressed)] *= scale
    return stressed


def main():
    mean, cov = load_inputs()
    rf_base = 0.075

    scenarios = [
        ("baseline", 1.00, 0.075),
        ("higher_volatility_20pct", 1.20, 0.075),
        ("higher_volatility_50pct", 1.50, 0.075),
        ("lower_volatility_20pct", 0.80, 0.075),
        ("risk_free_zero", 1.00, 0.000),
        ("risk_free_15pct", 1.00, 0.150),
        ("risk_free_25pct", 1.00, 0.250),
        ("returns_down_10pct", 1.00, None),
        ("returns_up_10pct", 1.00, None),
    ]

    results = []
    failures = []

    for name, cov_scale, rf in scenarios:
        scenario_mean = mean.copy()
        if name == "returns_down_10pct":
            scenario_mean *= 0.90
            rf = rf_base
        elif name == "returns_up_10pct":
            scenario_mean *= 1.10
            rf = rf_base

        scenario_cov = make_covariance(cov, cov_scale)

        try:
            max_sharpe = maximize_sharpe_ratio(
                scenario_mean, scenario_cov, rf
            )
            min_var = minimize_variance(scenario_mean, scenario_cov)

            max_check = check_weights(max_sharpe)
            min_check = check_weights(min_var)

            frontier = generate_efficient_frontier(
                scenario_mean, scenario_cov, points=30
            )

            frontier_ok = (
                len(frontier["returns"]) >= 20
                and len(frontier["returns"]) == len(frontier["volatilities"])
                and np.all(np.isfinite(frontier["volatilities"]))
            )

            max_metrics = portfolio_metrics(
                max_sharpe, scenario_mean, scenario_cov, rf
            )
            min_metrics = portfolio_metrics(
                min_var, scenario_mean, scenario_cov, rf
            )

            ok = (
                max_check["finite"]
                and max_check["sum_ok"]
                and max_check["bounds_ok"]
                and min_check["finite"]
                and min_check["sum_ok"]
                and min_check["bounds_ok"]
                and frontier_ok
                and np.isfinite(max_metrics["volatility"])
                and np.isfinite(min_metrics["volatility"])
            )

            if not ok:
                failures.append(name)

            results.append(
                {
                    "scenario": name,
                    "covariance_diagonal_scale": cov_scale,
                    "risk_free_rate": rf,
                    "max_sharpe_return": max_metrics["return"],
                    "max_sharpe_volatility": max_metrics["volatility"],
                    "max_sharpe_sharpe": max_metrics["sharpe"],
                    "max_sharpe_weight_sum": max_check["sum"],
                    "max_sharpe_max_weight": max_check["max_weight"],
                    "min_variance_return": min_metrics["return"],
                    "min_variance_volatility": min_metrics["volatility"],
                    "min_variance_weight_sum": min_check["sum"],
                    "min_variance_max_weight": min_check["max_weight"],
                    "frontier_points": int(len(frontier["returns"])),
                    "passed": ok,
                }
            )
        except Exception as exc:
            failures.append(name)
            results.append(
                {
                    "scenario": name,
                    "covariance_diagonal_scale": cov_scale,
                    "risk_free_rate": rf,
                    "passed": False,
                    "error": str(exc),
                }
            )

    report = {
        "test": "Markowitz optimiser stress test",
        "assets": list(mean.index),
        "scenario_count": len(scenarios),
        "passed_scenarios": len(scenarios) - len(failures),
        "failed_scenarios": failures,
        "all_passed": len(failures) == 0,
        "checks": [
            "weights are finite",
            "weights sum to 1",
            "long-only bounds are respected",
            "optimisation returns usable portfolio metrics",
            "efficient frontier returns at least 20 of 30 requested points",
        ],
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(REPORTS / "stress_test_results.csv", index=False)
    with (REPORTS / "stress_test_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== OPTIMISER STRESS TEST ===")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Passed: {report['passed_scenarios']}")
    print(f"Failed: {report['failed_scenarios'] or 'None'}")

    if failures:
        raise RuntimeError(
            "One or more optimiser stress scenarios failed. "
            "Review outputs/reports/stress_test_report.json."
        )


if __name__ == "__main__":
    main()
