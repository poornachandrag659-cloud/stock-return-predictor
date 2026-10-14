from __future__ import annotations
import numpy as np
import pandas as pd
from .backtest import performance_metrics

def block_bootstrap_paths(returns, n_paths=1000, block_size=5, seed=42) -> np.ndarray:
    """Circular moving-block bootstrap preserves some short-range dependence."""
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2: raise ValueError("At least two finite returns are required")
    if n_paths < 1 or block_size < 1: raise ValueError("n_paths and block_size must be positive")
    rng = np.random.default_rng(seed); n = len(x)
    blocks_needed = int(np.ceil(n / block_size))
    out = np.empty((n_paths, n), dtype=float)
    for i in range(n_paths):
        starts = rng.integers(0, n, size=blocks_needed)
        sampled = np.concatenate([x[(s + np.arange(block_size)) % n] for s in starts])[:n]
        out[i] = sampled
    return out

def monte_carlo_summary(returns, n_paths=1000, block_size=5, seed=42, initial_capital=10_000.0) -> tuple[pd.DataFrame, dict]:
    paths = block_bootstrap_paths(returns, n_paths, block_size, seed)
    rows=[]
    for i, r in enumerate(paths):
        m=performance_metrics(r)
        equity=initial_capital*np.cumprod(1+r)
        m.update({"path":i, "ending_equity":float(equity[-1])})
        rows.append(m)
    frame=pd.DataFrame(rows)
    keys=["total_return","cagr","annualized_volatility","sharpe","max_drawdown","ending_equity"]
    summary={}
    for key in keys:
        vals=frame[key].replace([np.inf,-np.inf],np.nan).dropna()
        summary[key]={"p05":float(vals.quantile(.05)),"median":float(vals.quantile(.50)),"p95":float(vals.quantile(.95))}
    summary["probability_of_loss"] = float((frame["total_return"] < 0).mean())
    summary["probability_of_underperforming_buy_hold"] = None
    return frame, summary
