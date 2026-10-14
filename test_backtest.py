import numpy as np
from stock_predictor.backtest import backtest, performance_metrics
from stock_predictor.monte_carlo import block_bootstrap_paths

def test_costs_reduce_returns():
    pred=np.array([.1,.1,-.1]); actual=np.array([.01,.02,-.01])
    bt=backtest(pred,actual,transaction_cost_bps=5,short_borrow_bps_per_day=1)
    assert (bt.strategy_return <= bt.position*actual + 1e-12).all()
    assert len(bt)==3
def test_metrics_known_values():
    m=performance_metrics(np.array([.01,.01]))
    assert np.isclose(m["total_return"],1.01**2-1)
    assert m["observations"]==2
def test_bootstrap_shape_and_seed():
    x=np.array([-.01,.0,.01,.02,-.02])
    a=block_bootstrap_paths(x,20,2,42); b=block_bootstrap_paths(x,20,2,42)
    assert a.shape==(20,5); assert np.array_equal(a,b)
