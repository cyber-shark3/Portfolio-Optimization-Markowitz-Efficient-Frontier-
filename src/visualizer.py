from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def plot_efficient_frontier(frontier, max_sharpe, min_vol, mean_returns, cov_matrix):
    from src.optimizer import portfolio_metrics
    out = ROOT / "outputs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    max_m = portfolio_metrics(max_sharpe, mean_returns, cov_matrix, 0.0)
    min_m = portfolio_metrics(min_vol, mean_returns, cov_matrix, 0.0)
    plt.figure(figsize=(10, 6))
    plt.plot(frontier["volatilities"] * 100, frontier["returns"] * 100,
             linewidth=2, label="Efficient Frontier")
    plt.scatter(max_m["volatility"] * 100, max_m["return"] * 100,
                s=80, label="Maximum Sharpe")
    plt.scatter(min_m["volatility"] * 100, min_m["return"] * 100,
                s=80, label="Minimum Variance")
    plt.xlabel("Annualised Volatility (%)")
    plt.ylabel("Annualised Return (%)")
    plt.title("Markowitz Efficient Frontier")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "efficient_frontier.png", dpi=300)
    plt.close()
