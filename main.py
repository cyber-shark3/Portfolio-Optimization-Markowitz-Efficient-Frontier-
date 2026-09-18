import json
from pathlib import Path
import pandas as pd
from src.data_loader import load_config, get_historical_returns
from src.optimizer import maximize_sharpe_ratio, minimize_variance, generate_efficient_frontier, portfolio_metrics
from src.visualizer import plot_efficient_frontier

ROOT = Path(__file__).resolve().parent

def main():
    print("Loading configuration...")
    config = load_config()
    print("Fetching prices and calculating returns/covariance...")
    mean_returns, cov_matrix, prices = get_historical_returns(config)
    rf = config["parameters"]["risk_free_rate"]

    print("Running maximum-Sharpe optimisation...")
    max_sharpe = maximize_sharpe_ratio(mean_returns, cov_matrix, rf)
    print("Running minimum-variance optimisation...")
    min_var = minimize_variance(mean_returns, cov_matrix)
    print("Building efficient frontier...")
    frontier = generate_efficient_frontier(mean_returns, cov_matrix)

    tickers = list(mean_returns.index)
    weights = pd.DataFrame({"Max Sharpe": max_sharpe, "Min Variance": min_var}, index=tickers)
    reports = ROOT / "outputs" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    weights.to_csv(reports / "optimal_weights.csv")

    metrics = {
        "max_sharpe": portfolio_metrics(max_sharpe, mean_returns, cov_matrix, rf),
        "min_variance": portfolio_metrics(min_var, mean_returns, cov_matrix, rf),
        "observations": int(len(prices)),
        "assets": tickers,
        "risk_free_rate": rf,
    }
    with (reports / "portfolio_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_efficient_frontier(frontier, max_sharpe, min_var, mean_returns, cov_matrix)
    print("\n=== RESULTS ===")
    print(weights.round(4))
    print("\nMax Sharpe:", {k: round(v, 4) for k, v in metrics["max_sharpe"].items()})
    print("Min Variance:", {k: round(v, 4) for k, v in metrics["min_variance"].items()})
    print("\nDone. Files saved under data/ and outputs.")

if __name__ == "__main__":
    main()
