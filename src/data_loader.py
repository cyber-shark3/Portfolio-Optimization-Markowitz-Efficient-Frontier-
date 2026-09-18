from pathlib import Path
import json

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
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if data.empty:
        raise RuntimeError("yfinance returned no price data.")

    if isinstance(data.columns, pd.MultiIndex):
        field = "Close" if "Close" in data.columns.get_level_values(0) else data.columns.levels[0][0]
        data = data[field]
    else:
        field = "Close" if "Close" in data.columns else data.columns[0]
        data = data[[field]].rename(columns={field: tickers[0]})

    return data.reindex(columns=tickers).sort_index()


def get_historical_returns(config: dict):
    tickers = config["portfolio"]["tickers"]
    start_date = config["portfolio"]["start_date"]
    end_date = config["portfolio"]["end_date"]
    trading_days = config["parameters"]["trading_days"]
    quality = config.get("data_quality", {})
    suspicious_limit = float(quality.get("suspicious_daily_return_abs", 1.0))
    min_observations = int(quality.get("min_observations", 252))

    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    reports_dir = ROOT / "outputs" / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching data for: {', '.join(tickers)}")
    prices = _download_prices(tickers, start_date, end_date)
    prices.to_csv(raw_dir / "historical_prices.csv")

    available = {ticker: int(prices[ticker].notna().sum()) for ticker in tickers}
    returns = prices.pct_change(fill_method=None)

    suspicious = returns.abs() > suspicious_limit
    suspicious_count = suspicious.sum().astype(int)

    suspicious_events = []
    for ticker in tickers:
        for timestamp in returns.index[suspicious[ticker].fillna(False)]:
            suspicious_events.append(
                {
                    "date": timestamp.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "daily_return": float(returns.at[timestamp, ticker]),
                }
            )

    # Do not silently repair extreme observations. Remove affected dates from the
    # optimisation sample and keep a report so every exclusion is visible.
    clean_returns = returns.mask(suspicious)
    clean_returns = clean_returns.dropna(how="any")

    valid_columns = clean_returns.columns[clean_returns.notna().sum() >= min_observations]
    clean_returns = clean_returns[valid_columns]

    if clean_returns.empty or len(clean_returns.columns) < 2:
        raise RuntimeError(
            "Not enough usable return data after quality checks. "
            "Review outputs/reports/data_quality_report.json."
        )

    prices_used = prices.loc[clean_returns.index, clean_returns.columns]

    mean_returns = clean_returns.mean() * trading_days
    cov_matrix = clean_returns.cov() * trading_days

    quality_report = {
        "source": "Yahoo Finance via yfinance",
        "price_mode": "auto_adjust=True",
        "start_date": start_date,
        "end_date": end_date,
        "suspicious_daily_return_abs_threshold": suspicious_limit,
        "minimum_observations_required": min_observations,
        "assets_requested": tickers,
        "assets_used": list(clean_returns.columns),
        "raw_price_observations": available,
        "suspicious_return_counts": {
            ticker: int(suspicious_count.get(ticker, 0)) for ticker in tickers
        },
        "excluded_dates_from_optimisation": int(len(returns) - len(clean_returns)),
        "usable_return_observations": int(len(clean_returns)),
        "suspicious_events": suspicious_events,
    }

    with (reports_dir / "data_quality_report.json").open("w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    clean_returns.to_csv(processed_dir / "daily_returns.csv")
    mean_returns.to_csv(processed_dir / "annualised_mean_returns.csv", header=["annual_return"])
    cov_matrix.to_csv(processed_dir / "annualised_covariance.csv")

    return mean_returns, cov_matrix, prices_used
