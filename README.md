# Markowitz Portfolio Optimisation — Efficient Frontier

A modular Python project for mean-variance portfolio optimisation using historical market data.

## What it does

- Downloads historical prices with yfinance.
- Caches raw prices and processed returns.
- Calculates annualised expected returns and covariance.
- Finds maximum-Sharpe and minimum-variance portfolios.
- Builds an efficient frontier.
- Saves weights, metrics and charts.

## Default portfolio

- OMU.JO
- NPK.JO
- SBK.JO
- TTO.JO
- FSR.JO

Ticker availability, liquidity and data quality must be checked before using results for real decisions. Forward-filling missing prices can affect measured returns and covariance.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## Method

For weights **w**, expected returns **μ**, and covariance matrix **Σ**:

- Expected return = **w'μ**
- Portfolio variance = **w'Σw**
- Volatility = **sqrt(w'Σw)**
- Sharpe = **(return − risk-free rate) / volatility**

The optimiser is long-only:

```
0 <= weight_i <= 1
sum(weight_i) = 1
```

This is a research/education project, not financial advice.
