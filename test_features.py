import numpy as np
import pandas as pd
from stock_predictor.features import build_features, FEATURE_COLUMNS, rsi

def sample_prices(n=180):
    rng=np.random.default_rng(5); returns=rng.normal(.0003,.01,n); close=100*np.cumprod(1+returns)
    return pd.DataFrame({"Open":close*(1+rng.normal(0,.002,n)),"High":close*1.01,"Low":close*.99,"Close":close,"Volume":rng.integers(1000,10000,n)},index=pd.bdate_range("2020-01-01",periods=n))
def test_features_finite_and_target_shift():
    prices=sample_prices(); frame=build_features(prices)
    assert set(FEATURE_COLUMNS).issubset(frame.columns)
    assert np.isfinite(frame[FEATURE_COLUMNS].to_numpy()).all()
    first=frame.index[0]; i=prices.index.get_loc(first)
    expected=prices.Close.iloc[i+1]/prices.Close.iloc[i]-1
    assert np.isclose(frame.loc[first,"target_return"],expected)
def test_target_last_day_removed():
    frame=build_features(sample_prices())
    assert frame.index.max() < sample_prices().index.max()
def test_rsi_bounded():
    s=pd.Series(np.linspace(1,100,80))
    values=rsi(s).dropna()
    assert ((values>=0)&(values<=100)).all()
