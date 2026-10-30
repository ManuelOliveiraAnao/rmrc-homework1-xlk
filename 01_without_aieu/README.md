# Baseline model without AI-EU

This folder runs independently and does not require the AI-EU data file.

- `code/analysis.py`: baseline model on the common sample of 2,971 trading dates.
- `code/estimation.py`: estimation without index parameters.
- `code/original_analysis.py`: complete original pipeline, with paths adapted
  to this structure. It uses 2,972 observations.
- `data/`: original prices, returns and earnings data.
- `results/common_sample/`: coefficients, filtered series, diagnostics and
  holdout evaluation for comparison with the index extensions.
- `results/original_sample/`: all tables and logs from the original project.
- `figures/`: figures separated by sample.

Run from this folder:

```bash
pip install -r requirements.txt
python code/analysis.py
```

To rerun the complete original pipeline: `python code/original_analysis.py`.
The common sample covers January 5, 2015 to October 27, 2026. Holdout estimation
uses the first 2,376 observations; evaluation uses the last 595 observations.
Evaluation is conditional on observed market and VIX values and retains the
original earnings labels. It is not a real-time trading forecast.
