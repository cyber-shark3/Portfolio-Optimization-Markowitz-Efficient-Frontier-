from pathlib import Path
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]

def load_config() -> dict:
    config_path = ROOT / "config" / "settings.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        import yaml
        return yaml.safe_load(f)

def _download_prices(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    data = yf.download(
        tickers, start=start_date, end=end_date,
        auto_adjust=False, progress=False, threads=True
    )
    if data.empty:
        raise RuntimeError("yfinance returned no price data.")
    if isinstance(data.columns, pd.MultiIndex):
        field = "Adj Close" if "Adj Close" in data.columns.get_level_values(0) else "Close"
        data = data[field]
    else:
        field = "Adj Close" if "Adj Close" in data.columns else "Close"
        data = data[[field]].rename(columns={field: tickers[0]})
    return data.reindex(columns=tickers).sort_index().ffill().dropna(how="all")

def get_historical_returns(config: dict):
    tickers = config["portfolio"]["tickers"]
    start_date = config["portfolio"]["start_date"]
    end_date = config["portfolio"]["end_date"]
    trading_days = config["parameters"]["trading_days"]

    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching data for: {', '.join(tickers)}")
    prices = _download_prices(tickers, start_date, end_date)
    prices.to_csv(raw_dir / "historical_prices.csv")

    returns = prices.pct_change().dropna(how="all")
    valid_columns = returns.columns[returns.notna().sum() > 1]
    returns = returns[valid_columns].dropna(how="any")
    prices = prices[valid_columns].loc[returns.index]

    if returns.empty or len(returns.columns) < 2:
        raise RuntimeError("Not enough usable return data for optimisation.")

    mean_returns = returns.mean() * trading_days
    cov_matrix = returns.cov() * trading_days

    returns.to_csv(processed_dir / "daily_returns.csv")
    mean_returns.to_csv(processed_dir / "annualised_mean_returns.csv", header=["annual_return"])
    cov_matrix.to_csv(processed_dir / "annualised_covariance.csv")
    return mean_returns, cov_matrix, prices
