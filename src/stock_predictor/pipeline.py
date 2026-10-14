from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from .data import download_prices
from .features import build_features, FEATURE_COLUMNS
from .model import fit_predict, prediction_metrics
from .backtest import backtest, performance_metrics
from .monte_carlo import monte_carlo_summary

def run_experiment(ticker="SPY", start="2015-01-01", end=None, test_fraction=.20, validation_fraction=.15,
                   initial_capital=10000, transaction_cost_bps=5, short_borrow_bps_per_day=1, threshold=0,
                   monte_carlo_paths=1000, monte_carlo_block_size=5, random_seed=42, model_name="random_forest",
                   output_dir="reports", prices=None):
    if not 0 < test_fraction < .5 or not 0 < validation_fraction < .5 or test_fraction + validation_fraction >= .8:
        raise ValueError("Use positive validation/test fractions whose sum is below 0.8")
    raw = prices.copy() if prices is not None else download_prices(ticker, start, end)
    dataset = build_features(raw)
    n=len(dataset); test_n=max(1,int(np.ceil(n*test_fraction))); trainval_n=n-test_n
    val_n=max(1,int(np.ceil(n*validation_fraction)))
    if trainval_n <= val_n + 50: raise ValueError("Not enough rows after feature engineering for chronological train/validation/test splits")
    train=dataset.iloc[:trainval_n-val_n]; validation=dataset.iloc[trainval_n-val_n:trainval_n]; test=dataset.iloc[trainval_n:]
    # Chronological validation is reported for transparency; final model is fitted only on pre-test rows.
    val_model, val_pred=fit_predict(train, validation, model_name, random_seed)
    validation_metrics=prediction_metrics(validation["target_return"],val_pred)
    final_model, predictions=fit_predict(dataset.iloc[:trainval_n], test, model_name, random_seed)
    pred_metrics=prediction_metrics(test["target_return"],predictions)
    bt=backtest(predictions,test["target_return"],initial_capital,transaction_cost_bps,short_borrow_bps_per_day,threshold)
    bt.index=test.index
    strategy_metrics=performance_metrics(bt["strategy_return"])
    buy_hold_metrics=performance_metrics(bt["buy_hold_return"])
    mc_paths, mc_summary=monte_carlo_summary(bt["strategy_return"],monte_carlo_paths,monte_carlo_block_size,random_seed,initial_capital)
    mc_summary["probability_of_underperforming_buy_hold"] = float((mc_paths["total_return"] < buy_hold_metrics["total_return"]).mean())
    # Save feature importance when the estimator exposes it.
    importance=[]
    if hasattr(final_model,"feature_importances_"):
        importance=sorted([{"feature":f,"importance":float(v)} for f,v in zip(FEATURE_COLUMNS,final_model.feature_importances_)],key=lambda z:z["importance"],reverse=True)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    bt.to_csv(out/"backtest.csv",index_label="date")
    mc_paths.to_csv(out/"monte_carlo_paths.csv",index=False)
    pred_frame=pd.DataFrame({"actual_return":test["target_return"],"predicted_return":predictions},index=test.index)
    pred_frame.to_csv(out/"predictions.csv",index_label="date")
    result={"ticker":ticker,"model":model_name,"start":str(raw.index.min().date()),"end":str(raw.index.max().date()),
        "raw_rows":int(len(raw)),"usable_rows":int(n),"split":{"train":int(len(train)),"validation":int(len(validation)),"test":int(len(test)),"chronological":True},
        "features":FEATURE_COLUMNS,"validation_prediction_metrics":validation_metrics,"test_prediction_metrics":pred_metrics,
        "strategy_metrics":strategy_metrics,"buy_hold_metrics":buy_hold_metrics,"monte_carlo":mc_summary,"feature_importance":importance,
        "assumptions":{"initial_capital":initial_capital,"transaction_cost_bps_per_unit_turnover":transaction_cost_bps,
            "short_borrow_bps_per_short_day":short_borrow_bps_per_day,"signal":"prediction sign at close t, held for next close-to-close return",
            "position":"daily rebalanced long/short/flat, gross exposure <= 1x","bootstrap":"circular moving-block bootstrap of realized test strategy returns"},
        "warning":"Research prototype, not investment advice. Historical backtests do not imply future performance."}
    (out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=True))
    _plot_equity(bt,out/"equity_curve.png",ticker)
    _plot_drawdown(bt,out/"drawdown.png")
    if importance: _plot_importance(importance,out/"feature_importance.png")
    _plot_mc(mc_paths,out/"monte_carlo_ending_equity.png",initial_capital)
    return result

def _plot_equity(bt,path,ticker):
    fig,ax=plt.subplots(figsize=(11,5)); ax.plot(bt.index,bt.strategy_equity,label="ML long/short (net)"); ax.plot(bt.index,bt.buy_hold_equity,label="Buy and hold",alpha=.85); ax.set_title(f"Out-of-sample equity — {ticker}"); ax.set_ylabel("Equity ($)"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)
def _plot_drawdown(bt,path):
    fig,ax=plt.subplots(figsize=(11,4)); ax.plot(bt.index,bt.strategy_drawdown,label="Strategy"); ax.plot(bt.index,bt.buy_hold_drawdown,label="Buy and hold",alpha=.8); ax.set_title("Out-of-sample drawdown"); ax.set_ylabel("Drawdown"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)
def _plot_importance(rows,path):
    top=rows[:12][::-1]; fig,ax=plt.subplots(figsize=(9,5)); ax.barh([r["feature"] for r in top],[r["importance"] for r in top]); ax.set_title("Random forest feature importance (descriptive, not causal)"); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)
def _plot_mc(frame,path,initial_capital):
    fig,ax=plt.subplots(figsize=(9,5)); ax.hist(frame["ending_equity"],bins=40); ax.axvline(initial_capital,linestyle="--",label="Initial capital"); ax.set_title("Bootstrap distribution of ending equity"); ax.set_xlabel("Ending equity ($)"); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)
