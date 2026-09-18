import numpy as np
from scipy.optimize import minimize

def _portfolio_return(weights, mean_returns):
    return float(np.dot(weights, mean_returns))

def _portfolio_volatility(weights, cov_matrix):
    variance = float(weights.T @ cov_matrix @ weights)
    return float(np.sqrt(max(variance, 0.0)))

def maximize_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    n = len(mean_returns)
    initial = np.full(n, 1.0 / n)
    def negative_sharpe(weights):
        vol = _portfolio_volatility(weights, cov_matrix)
        if vol <= 1e-12:
            return 1e6
        return -(_portfolio_return(weights, mean_returns) - risk_free_rate) / vol
    result = minimize(
        negative_sharpe, initial, method="SLSQP",
        bounds=tuple((0.0, 1.0) for _ in range(n)),
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Maximum-Sharpe optimisation failed: {result.message}")
    return result.x

def minimize_variance(mean_returns, cov_matrix):
    n = len(mean_returns)
    initial = np.full(n, 1.0 / n)
    result = minimize(
        lambda w: _portfolio_volatility(w, cov_matrix) ** 2,
        initial, method="SLSQP",
        bounds=tuple((0.0, 1.0) for _ in range(n)),
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Minimum-variance optimisation failed: {result.message}")
    return result.x

def generate_efficient_frontier(mean_returns, cov_matrix, points=100):
    targets = np.linspace(float(np.min(mean_returns)), float(np.max(mean_returns)), points)
    n = len(mean_returns)
    initial = np.full(n, 1.0 / n)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    returns, vols, weights = [], [], []
    for target in targets:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mean_returns) - t},
        ]
        result = minimize(
            lambda w: _portfolio_volatility(w, cov_matrix) ** 2,
            initial, method="SLSQP", bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if result.success:
            returns.append(target)
            vols.append(_portfolio_volatility(result.x, cov_matrix))
            weights.append(result.x)
    return {"returns": np.array(returns), "volatilities": np.array(vols), "weights": np.array(weights)}

def portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate):
    ret = _portfolio_return(weights, mean_returns)
    vol = _portfolio_volatility(weights, cov_matrix)
    sharpe = (ret - risk_free_rate) / vol if vol > 1e-12 else np.nan
    return {"return": ret, "volatility": vol, "sharpe": sharpe}
