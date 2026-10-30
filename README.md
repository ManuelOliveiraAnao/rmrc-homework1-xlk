# RMRC Homework I — XLK, earnings and AI-EU

Analysis of XLK ETF returns and volatility using CAPM and GJR-GARCH(1,1)-M.
The selected earnings announcements are those of Apple (AAPL), Microsoft
(MSFT) and Nvidia (NVDA), controlling for the S&P 500 and VIX. The extension
tests changes in the [AI-EU](https://www.policyuncertainty.com/aieu_daily.html)
index of newspaper coverage of artificial intelligence, economics and uncertainty.

## Project structure

| Folder | Code and results |
|---|---|
| [`01_without_aieu/`](01_without_aieu/README.md) | Baseline without AI-EU; includes the original project and common-sample estimates |
| [`02_with_aieu/`](02_with_aieu/README.md) | AI-EU extensions in the mean, variance and both equations |
| [`03_comparison/`](03_comparison/README.md) | Comparison tables and figures, read from the two separate versions |

Each version has its own `code/`, `data/`, `results/` and `figures/` folders.
Folders 01 and 02 can run independently. The original project files are
preserved under `01_without_aieu/`.

## Run the comparison

From the repository root:

```bash
pip install -r requirements.txt
python 01_without_aieu/code/analysis.py
python 02_with_aieu/code/analysis.py
python 03_comparison/code/compare.py
python verify_organization.py
```

To rerun the complete original pipeline with 2,972 observations:

```bash
python 01_without_aieu/code/original_analysis.py
```

Each version also has its own `requirements.txt`. Input data are cached
locally. The extension uses the included AI-EU snapshot rather than replacing
it with a newer download when running the analysis.

## Samples and interpretation

- Original project: January 2, 2015 to October 27, 2026; 2,972 daily observations.
- Comparison: January 5, 2015 to October 27, 2026; the same 2,971 observations in all four models.
- Holdout training sample: the first 2,376 observations.
- Holdout evaluation: the last 595 observations, June 13, 2024 to October 27, 2026.

AI-EU enters as the change in `log(1 + index)`, lagged by one calendar day.
The variance equation uses the magnitude of that change. The index level
shows evidence of nonstationarity. Absolute changes also reject KPSS level
stationarity, so the variance extension is exploratory.

In the tested specifications, AI-EU does not provide a convincing improvement:
BIC favors the baseline, and the small AIC advantage in the variance extension
does not translate into a lower holdout variance loss. Loss differences are
descriptive; their statistical significance has not been tested.

Evaluation is conditional on observed contemporaneous S&P 500 and VIX values.
It retains the original earnings classifications, which use full-sample
medians. AI-EU data are subject to revisions. These results do not establish
causality and are not a real-time trading backtest.

The individual README files describe each model and its files. Full extension
methodology is documented in [`03_comparison/METHODOLOGY.md`](03_comparison/METHODOLOGY.md).

## Sources

- Prices and earnings: Yahoo Finance, as in the original project.
- AI-EU: [Economic Policy Uncertainty](https://www.policyuncertainty.com/aieu_daily.html),
  series `daily_aieu_index`, snapshot downloaded October 30, 2026. CC BY 4.0 license.
- Validation: [`organization_validation.json`](organization_validation.json)
  and `python verify_organization.py`.
