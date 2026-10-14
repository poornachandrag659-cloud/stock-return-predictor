# Stock Return Predictor 📈

A reproducible quant research project that predicts **next-day close-to-close returns** from historical OHLCV features, evaluates a Random Forest against a simple baseline, backtests a daily long/short strategy after trading costs, and uses a **moving-block bootstrap** to show a distribution of outcomes rather than relying on one equity curve.

> **Important:** This is an educational research prototype, not financial advice or a trading signal. A backtest is not evidence of a durable edge. Shorting, slippage, taxes, market impact, borrow availability, survivorship, and data quality can materially change real-world results.

## What it does

- Downloads adjusted historical OHLCV prices through `yfinance` (internet required).
- Engineers lagged returns (1/2/5/10/21 days), SMA/EMA ratios, RSI, rolling volatility, volume z-score/change, intraday range, and close/open return.
- Labels each row with the **next** close-to-close return; features are only computed from data available at the current close.
- Uses chronological train / validation / test splits; never shuffles time-series rows. The final model is fitted on all pre-test observations and evaluated once on the held-out test period.
- Compares Random Forest, Extra Trees, Histogram Gradient Boosting, and a mean-return baseline.
- Backtests a sign-based long/short/flat strategy with turnover-based transaction costs and daily short-borrow cost, against buy-and-hold over the same dates.
- Reports MAE, RMSE, R², directional accuracy, total return, CAGR, annualized volatility, Sharpe, Sortino, max drawdown, and win rate.
- Applies a circular moving-block bootstrap to out-of-sample strategy returns to estimate 5th/50th/95th percentile outcome ranges and probability of loss / underperformance.
- Exports CSVs, JSON metrics, and PNG plots.

## Repository layout

```text
stock-return-predictor/
├── config.yaml
├── data/                  # downloaded data is not committed
├── reports/               # generated results are not committed
├── src/stock_predictor/
│   ├── data.py            # market data download
│   ├── features.py        # leakage-aware feature engineering
│   ├── model.py           # model definitions and prediction metrics
│   ├── backtest.py        # net returns and performance metrics
│   ├── monte_carlo.py     # moving-block bootstrap
│   ├── pipeline.py        # end-to-end experiment and plots
│   └── cli.py             # command-line interface
└── tests/
```

## Quick start (Windows PowerShell)

Install Python 3.10+ first. In PowerShell, move into the extracted repository directory and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
stock-predictor --config config.yaml
```

If PowerShell blocks activation, run this for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Alternative without activation:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\stock-predictor.exe --config config.yaml
```

## Run different experiments

```powershell
stock-predictor --ticker AAPL --start 2016-01-01 --model random_forest
stock-predictor --ticker MSFT --start 2016-01-01 --model extra_trees
stock-predictor --ticker SPY --start 2015-01-01 --model hist_gradient_boosting
stock-predictor --ticker SPY --start 2015-01-01 --model baseline
```

You can edit `config.yaml` or override ticker/start/end/model/output directory on the command line. `end_date: null` means download through the latest available date. Data availability depends on the Yahoo Finance service and ticker conventions.

## Output artifacts

| File | Description |
|---|---|
| `reports/summary.json` | Configuration, prediction metrics, strategy metrics, bootstrap quantiles, feature importance |
| `reports/predictions.csv` | Held-out actual and predicted next-day returns |
| `reports/backtest.csv` | Position, returns, costs, equity, drawdown |
| `reports/monte_carlo_paths.csv` | Metrics for each resampled return path |
| `reports/equity_curve.png` | Strategy vs buy-and-hold equity |
| `reports/drawdown.png` | Drawdown comparison |
| `reports/feature_importance.png` | Random Forest feature importance, if available |
| `reports/monte_carlo_ending_equity.png` | Bootstrap ending-equity distribution |

## Methodology and anti-leakage notes

1. **Label alignment:** row at date *t* has `target_return = Close[t+1] / Close[t] - 1`. The signal is assumed available after close *t*, then applied to that next close-to-close return.
2. **Chronology:** no shuffling. A pre-test validation window is reported; the final estimator trains on all observations strictly before the test window.
3. **Feature scope:** technical indicators use historical and current OHLCV only. The target column is never included in the feature matrix.
4. **Costs:** a position change of one unit incurs `transaction_cost_bps`; moving from -1 to +1 incurs two units of turnover. Short positions also pay the configured daily borrow proxy. These are simplified assumptions, not broker-specific execution simulation.
5. **Benchmark:** buy-and-hold uses the same test-period daily returns, without modeled transaction costs. This is a transparent benchmark, not necessarily an investable ETF implementation after fees.
6. **Bootstrap:** circular moving blocks preserve some local dependence; resampling still assumes the test return distribution is informative about future outcomes. It does not model regime changes and is not a guarantee or confidence interval for actual future returns.
7. **Repeated experiments:** repeatedly changing features, tickers, thresholds, and dates after inspecting test results causes data snooping. Reserve a final untouched period or use walk-forward evaluation before drawing conclusions.

### Known limitations / improvements

- Daily close data and simplified next-period fills; no intraday execution, market impact, taxes, dividends beyond adjusted-price behavior, or realistic borrow constraints.
- Single-asset model and single holdout period; use walk-forward evaluation and multiple market regimes for serious research.
- No hyperparameter search by design; if tuning, do it only within chronological training/validation windows.
- Long/short daily rebalancing can have high turnover. Cost assumptions are configurable and should be stress-tested.
- Feature importance is descriptive and should not be interpreted causally.
- LSTM is intentionally not included in the baseline implementation: tree models are easier to validate and provide a strong, auditable starting point. A sequence model should be added only with strict chronological validation.

## Suggested research extensions

1. Walk-forward expanding-window validation and purged/embargoed splits where labels overlap.
2. Cost sensitivity grid (0, 5, 10, 25 bps), threshold sweep performed on validation only.
3. Compare a linear ridge model and probabilistic directional model.
4. Add SPY/QQQ/sector ETFs, but avoid survivorship bias when testing individual-stock universes.
5. Add risk limits, volatility targeting, and position sizing decided only on past data.
6. Record data snapshots, package versions, and random seeds for reproducibility.

## Research references

- Gu, Kelly, and Xiu (2020), *Empirical Asset Pricing via Machine Learning*, Review of Financial Studies. https://academic.oup.com/rfs/article-abstract/33/5/2223/5758276
- López de Prado (2018), *The 10 Reasons Most Machine Learning Funds Fail*. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3104816
- Stefan Jansen, *Machine Learning for Trading* (open-source book/repository). https://github.com/stefan-jansen/machine-learning-for-trading
- QuantStart, *Forecasting Financial Time Series — Part 1*. https://www.quantstart.com/articles/Forecasting-Financial-Time-Series-Part-1/

## License

MIT. See [LICENSE](LICENSE).
