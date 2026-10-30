# XLK: comparison with the AI-EU index

This extension preserves the original financial data, analysis and results
under `01_without_aieu/`. Index extensions are stored in `02_with_aieu/`,
with cross-model comparisons in `03_comparison/`.

## Research question and findings

Does information about changes in newspaper coverage of AI, economics and
uncertainty improve a GJR-GARCH(1,1)-M model with earnings in the mean and variance?

The tested transformations and sample do not show a convincing improvement.
In the mean equation, the additional coefficient is -0.00798 percentage points
per standard deviation of the logarithmic index change, with robust p = 0.3949.
In the variance equation, absolute changes have a coefficient of 0.002275 and
an approximate robust p = 0.2210. The variance extension improves AIC by only
1.05 points, while worsening BIC by 4.95 points and holdout QLIKE loss by 0.62%.
Holdout differences are descriptive; their statistical significance has not
been established.

## Source and preparation

- Page: https://www.policyuncertainty.com/aieu_daily.html
- Official CSV: https://www.policyuncertainty.com/media/All_Daily_AIEU_Data.csv
- Selected series: `daily_aieu_index`, the intersection of AI, economics and uncertainty.
- The AI and AI-E indices are not used.
- The source measures US newspaper article frequency, scaled by the total
  number of articles. It is not a direct measure of investor sentiment or
  an indicator of exclusively negative news.
- License: CC BY 4.0. Attribution: Economic Policy Uncertainty / policyuncertainty.com.
- Snapshot downloaded October 30, 2026, with observations through October 29, 2026.
- One December 22, 2025 row was duplicated with identical values and removed
  before joining. No calendar dates were missing.
- Raw data: `02_with_aieu/data/All_Daily_AIEU_Data.csv`.
- Source hash and checks: `02_with_aieu/results/data_quality/source_and_checks.json`.

## Transformations and models

The AI-EU level fails stationarity diagnostics: ADF p approximately 0.506,
KPSS p <= 0.01. Taking its logarithm does not resolve the issue. The extension
therefore tests changes rather than the raw level:

```
c_t = log(1 + AI_EU_t) - log(1 + AI_EU_(t-1))
u_t = (c_(t-1) - mean(c_lag)) / std(c_lag)
v_t = abs(c_(t-1)) / std(abs(c_lag))
```

The `+1` handles genuine zeros without treating them as missing values.
The lag is one calendar day: Monday uses Sunday's change relative to Saturday,
not a weekend average. Lagging is applied before joining trading dates.
Insufficient history is not artificially filled. Signed changes pass ADF and
KPSS tests; absolute changes pass ADF but reject KPSS level stationarity.
The variance extension is therefore exploratory and may be unstable over time.

Equations (other variables retain their original definitions):

```
r_XLK,t = alpha + beta_SP*r_SP,t + beta_VIX*dVIX_t
          + d_good*GOOD_t + d_bad*BAD_t + lambda*sqrt(h_t)
          + theta*u_t + e_t
h_t = omega + (alpha_arch + gamma*I[e_(t-1)<0])*e_(t-1)^2
      + beta_garch*h_(t-1) + phi*NEWS_t + psi*v_t
```

Models: baseline (theta=psi=0), free theta, free psi, and both free.
Estimation uses Gaussian QML, robust sandwich standard errors and multiple
starting points. Restrictions ensure positive variances: omega>0,
alpha_arch>=0, alpha_arch+gamma>=0, beta_garch>=0, phi>=0, psi>=0 and
alpha_arch+gamma/2+beta_garch<1. The original code allowed negative phi;
the explicit restriction does not materially change its positive estimate.
Tests of psi=0 involve a null on the parameter-space boundary: the reported
Wald and reference LR p-values are approximations.

## Common-sample comparison

All models use 2,971 observations from January 5, 2015 to October 27, 2026.
January 2, 2015 is excluded from every model because two preceding calendar
days of AI-EU history are unavailable. This explains the small difference
in absolute AIC/BIC values relative to the original 2,972-observation summary.

| Model | AIC | BIC | Change in AIC | Change in BIC |
|---|---:|---:|---:|---:|
| Re-estimated baseline | 4814.35 | 4880.31 | 0 | 0 |
| AI-EU in mean | 4815.59 | 4887.55 | +1.25 | +7.24 |
| AI-EU in variance | 4813.30 | 4885.26 | -1.05 | +4.95 |
| AI-EU in both | 4814.43 | 4892.39 | +0.08 | +12.08 |

Lower AIC/BIC is preferable. BIC favors the baseline in every comparison.
AI-EU does not resolve residual ARCH effects or the OLS RESET rejection.
The positive GOOD coefficient is retained.

## Conditional holdout evaluation

The first 80% (2,376 observations through June 12, 2024) are used for estimation;
the last 20% (595 observations, June 13, 2024 to October 27, 2026) for evaluation.
Parameters and scaling remain fixed after training. Variance states update
using past errors, with the initial state computed from the training sample only.

| Model | Change in return MSE | Change in QLIKE loss |
|---|---:|---:|
| AI-EU in mean | -0.030% | +0.059% |
| AI-EU in variance | +0.214% | +0.623% |
| AI-EU in both | +0.186% | +0.697% |

Lower loss is preferable. All variance models use the same realized-variance
proxy: the baseline's squared holdout errors. This proxy is noisy and depends
on the baseline mean specification. The loss is
`q/h - log(max(q,1e-12)/h) - 1`. Gaussian log scores are also saved.

This is not a real-time backtest. Evaluation conditions on observed
contemporaneous S&P 500 and VIX values and retains the original earnings
classifications, which use full-sample medians. AI-EU is subject to historical
revisions, explicitly over the last 30 days. Lagging reduces the risk of using
articles published after the close but does not establish that the final
index value was available at that time. Results do not establish causality
or trading performance.

## Reproducibility and files

From the repository root:

```bash
pip install -r requirements.txt
python 01_without_aieu/code/analysis.py
python 02_with_aieu/code/analysis.py
python 03_comparison/code/compare.py
python verify_organization.py
```

- `03_comparison/results/model_comparison.csv`: AIC/BIC, diagnostics and convergence.
- Each model's `results/` subfolder: coefficients, uncertainty and filtered series.
- `03_comparison/results/holdout_comparison.csv`: holdout evaluation.
- `02_with_aieu/results/data_quality/stationarity.csv`: transformation tests.
- `03_comparison/results/ols_comparison.csv`: supplementary OLS comparison with HAC errors.
- Each version's `figures/` folder: its own figures; comparison figures are in folder 03.
- Each version's `run.log`: recorded model execution output.

Checks include returns independently recomputed from original prices with the
same date rule; unique joined dates; identical samples; convergence from
multiple starts; positive variances; non-decreasing nested-model likelihoods;
independently recomputed AIC/BIC and losses; and lag alignment on boundary dates
and a Monday. Generated figures are inspected for layout and labels.
