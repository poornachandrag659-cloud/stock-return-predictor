from __future__ import annotations
import numpy as np
import pandas as pd

def backtest(predictions, actual_returns, initial_capital=10_000.0, transaction_cost_bps=5.0,
             short_borrow_bps_per_day=1.0, threshold=0.0) -> pd.DataFrame:
    """Close-t signal is executed for the following close-to-close return.

    Positions are -1/0/+1 based on prediction sign and threshold. Turnover cost is
    charged on absolute position changes; first entry also incurs one unit of turnover.
    """
    pred = np.asarray(predictions, dtype=float).reshape(-1)
    actual = np.asarray(actual_returns, dtype=float).reshape(-1)
    if len(pred) != len(actual) or len(pred) == 0:
        raise ValueError("predictions and actual_returns must have equal non-zero lengths")
    position = np.where(pred > threshold, 1.0, np.where(pred < -threshold, -1.0, 0.0))
    previous = np.r_[0.0, position[:-1]]
    turnover = np.abs(position - previous)
    transaction_cost = turnover * (transaction_cost_bps / 10_000.0)
    borrow_cost = (position < 0).astype(float) * (short_borrow_bps_per_day / 10_000.0)
    strategy_return = position * actual - transaction_cost - borrow_cost
    buy_hold_return = actual.copy()
    dates = None
    result = pd.DataFrame({"prediction": pred, "actual_return": actual, "position": position,
        "turnover": turnover, "transaction_cost": transaction_cost, "borrow_cost": borrow_cost,
        "strategy_return": strategy_return, "buy_hold_return": buy_hold_return})
    result["strategy_equity"] = initial_capital * (1 + result["strategy_return"]).cumprod()
    result["buy_hold_equity"] = initial_capital * (1 + result["buy_hold_return"]).cumprod()
    result["strategy_drawdown"] = result["strategy_equity"] / result["strategy_equity"].cummax() - 1
    result["buy_hold_drawdown"] = result["buy_hold_equity"] / result["buy_hold_equity"].cummax() - 1
    return result

def performance_metrics(returns, periods_per_year=252) -> dict:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        raise ValueError("returns must contain finite observations")
    equity = np.cumprod(1 + r)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1
    total = float(equity[-1] - 1)
    years = len(r) / periods_per_year
    cagr = float(equity[-1] ** (1 / years) - 1) if equity[-1] > 0 and years > 0 else float("nan")
    vol = float(np.std(r, ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else 0.0
    mean = float(np.mean(r))
    std = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    sharpe = float(mean / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    downside = r[r < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    sortino = float(mean / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else float("nan")
    return {"observations": int(len(r)), "total_return": total, "cagr": cagr, "annualized_volatility": vol,
            "sharpe": sharpe, "sortino": sortino, "max_drawdown": float(np.min(drawdown)),
            "win_rate": float(np.mean(r > 0)), "best_day": float(np.max(r)), "worst_day": float(np.min(r))}
