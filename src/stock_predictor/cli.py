from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from .pipeline import run_experiment

def main():
    p=argparse.ArgumentParser(description="Leakage-aware stock return prediction and bootstrap backtest")
    p.add_argument("--config",default="config.yaml")
    p.add_argument("--ticker"); p.add_argument("--start"); p.add_argument("--end")
    p.add_argument("--model",choices=["random_forest","extra_trees","hist_gradient_boosting","baseline"])
    p.add_argument("--output-dir")
    args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text()) if Path(args.config).exists() else {}
    cfg.update({k:v for k,v in {"ticker":args.ticker,"start":args.start,"end":args.end,"model_name":args.model,"output_dir":args.output_dir}.items() if v is not None})
    result=run_experiment(**cfg)
    print("\n=== Out-of-sample prediction metrics ===")
    for k,v in result["test_prediction_metrics"].items(): print(f"{k:24s}: {v:.6f}")
    print("\n=== Strategy vs buy-and-hold ===")
    for name,key in [("ML strategy","strategy_metrics"),("Buy and hold","buy_hold_metrics")]:
        print(name)
        for k,v in result[key].items(): print(f"  {k:22s}: {v:.6f}" if isinstance(v,(float,int)) else f"  {k}: {v}")
    print("\n=== Monte Carlo (block bootstrap) ===")
    for k,v in result["monte_carlo"].items(): print(f"{k}: {v}")
    print(f"\nSaved outputs to: {cfg.get('output_dir','reports')}")
if __name__=="__main__": main()
