# Comparison of the two versions

Tables and figures containing multiple models are kept here.
`code/compare.py` reads results from folders 01 and 02 without re-estimating
the models. Run it after executing the individual model scripts:

```bash
python code/compare.py
python code/supplementary_tests.py
```

All models are compared on the same 2,971 observations. QLIKE uses the same
realized-variance proxy for every model: the baseline's squared holdout errors.
These loss comparisons are not trading backtests.

[`METHODOLOGY.md`](METHODOLOGY.md) documents the source, transformations,
specifications, results and limitations. The OLS and LR tables are supplementary
outputs from the extension analysis. The comparison script rebuilds AIC/BIC,
holdout losses and figures from the separate model results.
