# Models with AI-EU

This folder runs independently. Every model executed here includes AI-EU
information; the baseline without the index is estimated in folder 01.

- `code/analysis.py`: runs the three index extensions.
- `code/estimation.py`: AI-EU preparation and model estimation.
- `data/`: financial data, earnings and the official AI-EU snapshot.
- `results/ai_in_mean/`: signed AI-EU change in the return equation.
- `results/ai_in_variance/`: absolute AI-EU change in the variance equation.
- `results/ai_in_both/`: both additional components.
- `results/data_quality/`: joined data, source information and stationarity tests.
- `figures/`: individual model figures and the AI-EU time series.

Run from this folder:

```bash
pip install -r requirements.txt
python code/analysis.py
```

The models use changes in log(1+AI-EU), lagged by one calendar day, and the
same common sample of 2,971 trading dates as the baseline. Holdout scaling
parameters are estimated using only the first 80% of the data.
Source: [Economic Policy Uncertainty](https://www.policyuncertainty.com/aieu_daily.html)
(CC BY 4.0).

The index is subject to revisions. Absolute changes reject KPSS level
stationarity, so the variance extension is exploratory. The original earnings
labels are retained, and evaluation is conditional on observed market and
VIX values. It is not a real-time trading backtest.
