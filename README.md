# RMRC Homework I — Tech news and XLK (CAPM + GJR-GARCH(1,1)-M)

Impact of technology news (AAPL, MSFT, NVDA earnings announcements) on the return and conditional volatility of the XLK ETF, controlling for the S&P 500 and the VIX. Daily data, 2015-01-02 to 2026-10-27, from Yahoo Finance.

```bash
pip install -r requirements.txt
python analysis.py
```

- `data/`: raw prices, earnings dates and the final dataset (cached; delete to re-download)
- `results/`: CSV tables and full log (`run_log.txt`)
- `figures/`: prices/returns, conditional variance h_t, standardised residuals, news impact curve
