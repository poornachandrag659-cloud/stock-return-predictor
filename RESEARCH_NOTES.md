# Research notes: how to interpret this project

## Research question
Can lagged price/volume-derived features provide out-of-sample information about the next daily close-to-close return, and does a simple sign-based strategy retain value after stylized trading costs?

## Hypotheses
- H0: forecasts do not improve meaningfully over a mean-return baseline and do not generate positive net risk-adjusted performance.
- H1: forecasts improve out-of-sample prediction and generate net performance that is robust to cost assumptions and resampling.

A single positive test result is not enough to reject H0. Results should persist across multiple periods, assets, and conservative costs without extensive parameter search.

## Why Random Forest first?
Tree ensembles capture nonlinear interactions with limited feature scaling requirements and offer feature importance diagnostics. They are a reasonable benchmark, not a guarantee of superior performance. A naive LSTM can easily overfit a small noisy financial dataset.

## Minimum reporting checklist
- [ ] Record ticker, dates, data source, package versions, and configuration.
- [ ] Confirm split dates are strictly chronological.
- [ ] Compare against mean-return prediction baseline.
- [ ] Report test prediction metrics and net trading metrics separately.
- [ ] Include transaction and short-borrow costs; test more conservative cost scenarios.
- [ ] Report bootstrap quantiles and probability of loss / underperformance.
- [ ] Do not select the best strategy by repeatedly inspecting the test set.
- [ ] State limitations and avoid claims of guaranteed returns.

## Monte Carlo caveat
The bootstrap distribution answers: “What range of outcomes arises if blocks resembling the observed test returns are rearranged and resampled?” It does **not** answer: “What is the probability this strategy will make money in the real future?” The latter requires assumptions that are unlikely to be fully satisfied in markets.
