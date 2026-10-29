"""
Homework I - RMRC 13026 (Fall 2026)
Research question: do technology news (earnings announcements of XLK's
mega-cap constituents AAPL, MSFT, NVDA) affect the return and the conditional
volatility of the Technology Select Sector SPDR ETF (XLK), after controlling
for the market (S&P 500) and market fear (VIX)?

CAPM-type mean equation (Rf = 0):
    r_XLK,t = a + b*r_SP,t + bV*dlnVIX_t + d+*GOOD_t + d-*BAD_t + l*sqrt(h_t) + e_t
GJR-GARCH(1,1)-M-X variance equation:
    h_t = w + (al + g*1[e_{t-1}<0]) e_{t-1}^2 + be*h_{t-1} + phi*NEWS_t

Run:  python analysis.py      (outputs -> results/, figures/)
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from scipy import stats, optimize
from scipy.special import gammaln
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, linear_reset, acorr_breusch_godfrey
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.numdiff import approx_hess, approx_fprime

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RES, FIG = (os.path.join(HERE, d) for d in ("data", "results", "figures"))
for d in (DATA, RES, FIG):
    os.makedirs(d, exist_ok=True)

START, END = "2014-12-31", "2026-10-28"   # sample: 2015-01-02 .. 2026-10-27
LAGS = [5, 10, 20]
OUT = []                                   # text log written to results/summary.txt


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def section(t):
    log("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def lb_table(x, name):
    t = acorr_ljungbox(x, lags=LAGS)
    t.index = [f"{name} Q({l})" for l in LAGS]
    return t.rename(columns={"lb_stat": "Q", "lb_pvalue": "p-value"})


# ----------------------------------------------------------------------------
# 0. DATA
# ----------------------------------------------------------------------------
px_file = os.path.join(DATA, "prices_raw.csv")
if not os.path.exists(px_file):
    px = yf.download(["XLK", "^GSPC", "^VIX"], start=START, end=END,
                     auto_adjust=True, progress=False)["Close"]
    px.to_csv(px_file)
px = pd.read_csv(px_file, index_col=0, parse_dates=True).dropna()
px.columns = ["XLK", "SP500", "VIX"]

ret = pd.DataFrame({
    "r_XLK": 100 * np.log(px.XLK).diff(),
    "r_SP": 100 * np.log(px.SP500).diff(),
    "dVIX": 100 * np.log(px.VIX).diff(),
}).dropna()
tdays = ret.index

# Tech news: earnings announcements of the 3 largest XLK holdings.
ea_file = os.path.join(DATA, "earnings_dates.csv")
if not os.path.exists(ea_file):
    rows = []
    for tk in ["AAPL", "MSFT", "NVDA"]:
        e = yf.Ticker(tk).get_earnings_dates(limit=100)
        e = e.reset_index().rename(columns={"Earnings Date": "datetime"})
        e["ticker"] = tk
        rows.append(e)
    ea = pd.concat(rows)
    ea["datetime"] = pd.to_datetime(ea["datetime"], utc=True).dt.tz_convert("America/New_York")
    ea.to_csv(ea_file, index=False)
ea = pd.read_csv(ea_file)
ea["datetime"] = pd.to_datetime(ea["datetime"], utc=True).dt.tz_convert("America/New_York")
ea = ea.dropna(subset=["Reported EPS"])


def reaction_day(ts):
    """First trading day on which the market can react to the announcement."""
    d = pd.Timestamp(ts.date())
    after_close = ts.hour + ts.minute / 60 >= 9.5      # released during/after session
    pos = tdays.searchsorted(d, side="right" if after_close else "left")
    return tdays[pos] if pos < len(tdays) else pd.NaT


ea["react"] = ea["datetime"].apply(reaction_day)
ea = ea.dropna(subset=["react"])
ea = ea[(ea.react >= tdays[0]) & (ea.react <= tdays[-1])]
# Mega-caps almost always "beat" consensus, so a beat per se is expected news.
# GOOD = surprise above the firm's median surprise in the sample; BAD = below (incl. misses).
ea["good"] = (ea["Surprise(%)"] > ea.groupby("ticker")["Surprise(%)"].transform("median")).astype(int)
ea[["ticker", "datetime", "react", "EPS Estimate", "Reported EPS", "Surprise(%)", "good"]] \
    .sort_values("react").to_csv(os.path.join(RES, "news_events.csv"), index=False)

ret["NEWS"] = ret.index.isin(ea.react).astype(int)
ret["GOOD"] = ret.index.isin(ea.loc[ea.good == 1, "react"]).astype(int)
ret["BAD"] = ret.index.isin(ea.loc[ea.good == 0, "react"]).astype(int)
ret.to_csv(os.path.join(DATA, "dataset.csv"))

section("0. DATA")
log(f"Sample: {ret.index[0].date()} to {ret.index[-1].date()}, T = {len(ret)} daily obs")
log(f"Earnings events (AAPL, MSFT, NVDA): {len(ea)}; news days: {ret.NEWS.sum()} "
    f"(good = above-median surprise: {ret.GOOD.sum()}, bad = below-median: {ret.BAD.sum()}; "
    f"days with 2 releases may be both)")


def describe(df):
    d = pd.DataFrame({
        "mean": df.mean(), "std": df.std(), "min": df.min(), "max": df.max(),
        "skew": df.skew(), "ex.kurt": df.kurt(),
        "JB": [stats.jarque_bera(df[c])[0] for c in df],
        "JB p": [stats.jarque_bera(df[c])[1] for c in df],
    })
    d["ann.mean %"] = d["mean"] * 252
    d["ann.vol %"] = d["std"] * np.sqrt(252)
    return d


desc = describe(ret[["r_XLK", "r_SP", "dVIX"]])
desc.to_csv(os.path.join(RES, "t0_descriptives.csv"))
log(desc.round(3).to_string())
log("\nXLK return on news vs non-news days:")
log(ret.groupby("NEWS").r_XLK.agg(["count", "mean", "std", lambda x: x.abs().mean()])
    .rename(columns={"<lambda_0>": "mean|r|"}).round(3).to_string())

fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
px.XLK.plot(ax=ax[0], color="C0", title="XLK (adjusted close, USD)")
ret.r_XLK.plot(ax=ax[1], color="C0", lw=0.5, title="XLK daily log-return (%)")
px.VIX.plot(ax=ax[2], color="C3", title="VIX")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "f1_prices_returns.png"), dpi=150); plt.close()

# ----------------------------------------------------------------------------
# 1. STATIONARITY
# ----------------------------------------------------------------------------
section("1. STATIONARITY (ADF H0: unit root; KPSS H0: stationary) + Ljung-Box")
st = []
for name, x in [("ln P XLK", np.log(px.XLK)), ("ln P SP500", np.log(px.SP500)),
                ("VIX level", px.VIX), ("r_XLK", ret.r_XLK), ("r_SP", ret.r_SP),
                ("dlnVIX", ret.dVIX)]:
    reg = "ct" if name.startswith("ln P") else "c"
    a = adfuller(x.dropna(), regression=reg, autolag="AIC")
    k = kpss(x.dropna(), regression=reg, nlags="auto")
    st.append([name, reg, a[0], a[1], k[0], k[1]])
st = pd.DataFrame(st, columns=["series", "det.", "ADF", "ADF p", "KPSS", "KPSS p"]).set_index("series")
st.to_csv(os.path.join(RES, "t1_stationarity.csv"))
log(st.round(4).to_string())

lb1 = pd.concat([lb_table(ret.r_XLK, "r_XLK"), lb_table(ret.r_XLK ** 2, "r_XLK^2")])
lb1.to_csv(os.path.join(RES, "t1_ljungbox_returns.csv"))
log("\n" + lb1.round(4).to_string())

# ----------------------------------------------------------------------------
# 2-3. CONDITIONAL MEAN: OLS, RESET, autocorrelation, inference
# ----------------------------------------------------------------------------
section("2. CONDITIONAL MEAN SPECIFICATION (OLS)")
y = ret.r_XLK
specs = {
    "M1 CAPM": ["r_SP"],
    "M2 CAPM+VIX": ["r_SP", "dVIX"],
    "M3 CAPM+VIX+News": ["r_SP", "dVIX", "GOOD", "BAD"],
}
ols = {}
rows = []
for k, cols in specs.items():
    X = sm.add_constant(ret[cols])
    m = sm.OLS(y, X).fit()
    ols[k] = m
    rs = linear_reset(m, power=3, use_f=True)
    bg = acorr_breusch_godfrey(m, nlags=5)
    lb = acorr_ljungbox(m.resid, lags=[10])
    rows.append([k, m.rsquared, m.rsquared_adj, m.aic, m.bic, rs.fvalue, rs.pvalue,
                 lb.lb_stat.iloc[0], lb.lb_pvalue.iloc[0], bg[2], bg[3]])
spec_tab = pd.DataFrame(rows, columns=["model", "R2", "adj R2", "AIC", "BIC", "RESET F", "RESET p",
                                       "LB Q(10)", "LB p", "BG(5) F", "BG p"]).set_index("model")
spec_tab.to_csv(os.path.join(RES, "t2_mean_specification.csv"))
log(spec_tab.round(4).to_string())

# Residual autocorrelation -> check whether an AR(1) term is needed
X3 = sm.add_constant(ret[specs["M3 CAPM+VIX+News"]])
m3 = ols["M3 CAPM+VIX+News"]
Xar = X3.copy(); Xar["ar1"] = m3.resid.shift(1)
Xar = Xar.dropna()
m_ar = sm.OLS(y.loc[Xar.index], Xar).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
log("\nRESET robust to heteroskedasticity (M3): "
    + ", ".join(f"power {pw} {ct}: p = {linear_reset(m3, power=pw, use_f=True, cov_type=ct, cov_kwds=kw).pvalue:.4f}"
                for pw in (2, 3) for ct, kw in (("HC3", {}), ("HAC", {"maxlags": 10}))))
yb = ret.groupby(ret.index.year).apply(
    lambda g: sm.OLS(g.r_XLK, sm.add_constant(g[["r_SP", "dVIX"]])).fit().params["r_SP"])
yb.name = "beta_SP"
yb.to_csv(os.path.join(RES, "t2_beta_by_year.csv"))
log("\nYear-by-year beta_SP (parameter instability is a likely source of the RESET rejection):")
log(yb.round(3).to_frame().T.to_string())
Xtr = X3.copy(); Xtr["r_SP x trend"] = ret.r_SP * np.arange(len(ret)) / len(ret)
m_tr = sm.OLS(y, Xtr).fit()
m_trh = sm.OLS(y, Xtr).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
log(f"Trend-beta model: beta_SP = {m_trh.params['r_SP']:.3f} + {m_trh.params['r_SP x trend']:.3f}*(t/T) "
    f"(HAC p = {m_trh.pvalues['r_SP x trend']:.4g}); R2 = {m_tr.rsquared:.4f}; "
    f"RESET p = {linear_reset(m_tr, power=3, use_f=True).pvalue:.4f}")
log(f"\nAR(1) on residuals check: coef = {m_ar.params['ar1']:.4f}, HAC p = {m_ar.pvalues['ar1']:.4f}")

section("3. FIT AND INFERENCE (final OLS spec M3, HAC Newey-West SE)")
m3h = sm.OLS(y, X3).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
log(m3h.summary().as_text())
tests = {
    "H0: alpha = 0": "const = 0",
    "H0: beta_SP = 1": "r_SP = 1",
    "H0: beta_VIX = beta_SP": "dVIX - r_SP = 0",
    "H0: beta_VIX = 0": "dVIX = 0",
    "H0: d_good = d_bad = 0": "GOOD = 0, BAD = 0",
    "H0: d_good = d_bad": "GOOD - BAD = 0",
}
inf = []
for lab, h in tests.items():
    w = m3h.wald_test(h, scalar=True)
    inf.append([lab, float(w.statistic), float(w.pvalue)])
inf = pd.DataFrame(inf, columns=["hypothesis", "Wald chi2", "p-value"]).set_index("hypothesis")
coef_ols = pd.DataFrame({"coef": m3h.params, "HAC se": m3h.bse, "t": m3h.tvalues, "p": m3h.pvalues})
coef_ols.to_csv(os.path.join(RES, "t3_ols_coefficients.csv"))
inf.to_csv(os.path.join(RES, "t3_ols_wald_tests.csv"))
log("\n" + inf.round(4).to_string())
log(f"\nAnnualised alpha (OLS): {252 * m3h.params['const']:.2f}% per year")

# ----------------------------------------------------------------------------
# 4. ARCH EFFECTS IN OLS RESIDUALS
# ----------------------------------------------------------------------------
section("4. RESIDUAL DIAGNOSTICS: kurtosis, Ljung-Box on e^2, ARCH-LM")
e = m3.resid
log(f"Residual skewness = {stats.skew(e):.3f}, excess kurtosis = {stats.kurtosis(e):.3f}, "
    f"JB = {stats.jarque_bera(e)[0]:.1f} (p = {stats.jarque_bera(e)[1]:.4f})")
lb4 = pd.concat([lb_table(e, "e"), lb_table(e ** 2, "e^2")])
arch_lm = het_arch(e, nlags=5)
lb4.to_csv(os.path.join(RES, "t4_resid_ljungbox.csv"))
log(lb4.round(4).to_string())
log(f"ARCH-LM(5): LM = {arch_lm[0]:.2f}, p = {arch_lm[1]:.4g}")

# ----------------------------------------------------------------------------
# 5. LEVERAGE CHECK + GJR-GARCH(1,1)-M (custom QML estimation)
# ----------------------------------------------------------------------------
section("5. LEVERAGE CHECK AND GJR-GARCH(1,1)-M ESTIMATION")
c1 = np.corrcoef(ret.r_XLK.values[1:] ** 2, ret.r_XLK.values[:-1])[0, 1]
c2 = np.corrcoef(e.values[1:] ** 2, e.values[:-1])[0, 1]
se_c = 1 / np.sqrt(len(e))
log(f"corr(r_t^2, r_t-1) = {c1:.4f};  corr(e_t^2, e_t-1) = {c2:.4f}  (approx. 2*se = {2 * se_c:.4f})")
log("Negative values -> bad news raises next-day volatility more than good news (leverage effect).")

Y = ret.r_XLK.values
XM = np.column_stack([np.ones(len(ret)), ret.r_SP, ret.dVIX, ret.GOOD, ret.BAD])
ZV = ret.NEWS.values.astype(float)
MEAN_NAMES = ["alpha", "beta_SP", "beta_VIX", "d_good", "d_bad"]


def garch_filter(p, gjr=True, inmean=True, xvar=True, return_all=False, dist="normal"):
    k = XM.shape[1]
    b = p[:k]
    lam = p[k]
    w, a, g, bt, phi = p[k + 1:k + 6]
    if not gjr: g = 0.0
    if not inmean: lam = 0.0
    if not xvar: phi = 0.0
    T = len(Y)
    h = np.empty(T); eps = np.empty(T)
    xb = XM @ b
    h[0] = np.var(Y - xb)
    for t in range(T):
        if t > 0:
            ep = eps[t - 1]
            h[t] = w + (a + g * (ep < 0)) * ep * ep + bt * h[t - 1] + phi * ZV[t]
        if h[t] <= 1e-8:
            h[t] = 1e-8
        eps[t] = Y[t] - xb[t] - lam * np.sqrt(h[t])
    if dist == "t":                     # standardised Student-t, nu = p[k + 6]
        nu = p[k + 6]
        ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2) * h)
              - 0.5 * (nu + 1) * np.log1p(eps ** 2 / (h * (nu - 2))))
    else:
        ll = -0.5 * (np.log(2 * np.pi) + np.log(h) + eps ** 2 / h)
    if return_all:
        return ll, h, eps
    return ll


def fit_garch(gjr=True, inmean=True, xvar=True, label="", dist="normal"):
    b0 = np.linalg.lstsq(XM, Y, rcond=None)[0]
    p0 = np.r_[b0, 0.0, 0.02, 0.03, 0.08 if gjr else 0.0, 0.88, 0.05 if xvar else 0.0, 8.0]
    free = np.array([True] * (XM.shape[1]) + [inmean, True, True, gjr, True, xvar, dist == "t"])
    bnds = [(None, None)] * XM.shape[1] + [(-2, 2), (1e-6, 5), (0, 1), (-1, 1), (0, 0.9999), (-5, 50), (2.1, 100)]

    def nll(q):
        p = p0.copy(); p[free] = q
        w, a, g, bt = p[XM.shape[1] + 1:XM.shape[1] + 5]
        if a + g < 0 or a + 0.5 * g + bt >= 1:   # positivity / covariance stationarity
            return 1e10
        return -garch_filter(p, gjr, inmean, xvar, dist=dist).sum()

    bf = [b for b, f in zip(bnds, free) if f]
    r = optimize.minimize(nll, p0[free], method="L-BFGS-B", bounds=bf)
    r = optimize.minimize(nll, r.x, method="Nelder-Mead",
                          options={"maxiter": 20000, "maxfev": 20000, "xatol": 1e-7, "fatol": 1e-7})
    r = optimize.minimize(nll, r.x, method="L-BFGS-B", bounds=bf)
    p = p0.copy(); p[free] = r.x

    # Bollerslev-Wooldridge robust (QML sandwich) standard errors
    def ll_vec(q):
        pp = p0.copy(); pp[free] = q
        return garch_filter(pp, gjr, inmean, xvar, dist=dist)

    H = approx_hess(r.x, lambda q: ll_vec(q).sum())
    G = approx_fprime(r.x, ll_vec)
    Hinv = np.linalg.inv(-H)
    V = Hinv @ (G.T @ G) @ Hinv
    se = np.full(len(p), np.nan); se[free] = np.sqrt(np.diag(V))
    names = MEAN_NAMES + ["lambda (in-mean)", "omega", "alpha_arch", "gamma (GJR)", "beta_garch", "phi (news var)", "nu (t dof)"]
    tab = pd.DataFrame({"coef": p, "robust se": se}, index=names)
    tab["z"] = tab.coef / tab["robust se"]
    tab["p"] = 2 * (1 - stats.norm.cdf(np.abs(tab.z)))
    tab = tab[free]
    ll, h, eps = garch_filter(p, gjr, inmean, xvar, return_all=True, dist=dist)
    k = free.sum()
    info = dict(label=label, LL=ll.sum(), k=int(k), AIC=-2 * ll.sum() + 2 * k,
                BIC=-2 * ll.sum() + np.log(len(Y)) * k)
    return dict(p=p, tab=tab, V=V, free=free, h=h, eps=eps, z=eps / np.sqrt(h), info=info, names=names)


models = {
    "GARCH(1,1)": fit_garch(gjr=False, inmean=False, xvar=False, label="GARCH(1,1)"),
    "GJR-GARCH(1,1)": fit_garch(gjr=True, inmean=False, xvar=False, label="GJR-GARCH(1,1)"),
    "GJR-GARCH(1,1)-M": fit_garch(gjr=True, inmean=True, xvar=False, label="GJR-GARCH(1,1)-M"),
    "GJR-GARCH(1,1)-M + News in variance": fit_garch(gjr=True, inmean=True, xvar=True,
                                                     label="GJR-GARCH(1,1)-M-X"),
}
cmp_ = pd.DataFrame([m["info"] for m in models.values()]).set_index("label")
cmp_.to_csv(os.path.join(RES, "t5_model_comparison.csv"))
log(cmp_.round(2).to_string())

lr = []
pairs = [("GARCH(1,1)", "GJR-GARCH(1,1)", "gamma = 0"),
         ("GJR-GARCH(1,1)", "GJR-GARCH(1,1)-M", "lambda = 0"),
         ("GJR-GARCH(1,1)-M", "GJR-GARCH(1,1)-M + News in variance", "phi = 0")]
for a, b, h in pairs:
    stat = 2 * (models[b]["info"]["LL"] - models[a]["info"]["LL"])
    lr.append([f"{a} -> {b}", h, stat, 1 - stats.chi2.cdf(stat, 1)])
lr = pd.DataFrame(lr, columns=["nesting", "H0", "LR", "p-value"]).set_index("nesting")
lr.to_csv(os.path.join(RES, "t5_lr_tests.csv"))
log("\n" + lr.round(4).to_string())

for nm, m in models.items():
    m["tab"].to_csv(os.path.join(RES, f"t5_coef_{nm.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('+', 'plus')}.csv"))

FINAL = "GJR-GARCH(1,1)-M + News in variance"
fm = models[FINAL]
log(f"\nFINAL MODEL: {FINAL} (Gaussian QML, Bollerslev-Wooldridge robust SE)")
log(fm["tab"].round(4).to_string())
tm = fit_garch(gjr=True, inmean=True, xvar=True, label="GJR-GARCH(1,1)-M-X Student-t", dist="t")
tm["tab"].to_csv(os.path.join(RES, "t5_coef_final_student_t.csv"))
log(f"\nROBUSTNESS: same model with Student-t innovations (LL = {tm['info']['LL']:.2f}, "
    f"AIC = {tm['info']['AIC']:.2f}, BIC = {tm['info']['BIC']:.2f})")
log(tm["tab"].round(4).to_string())
log("\nBaseline GJR-GARCH(1,1)-M (no news in variance) for reference:")
log(models["GJR-GARCH(1,1)-M"]["tab"].round(4).to_string())

# Benchmark: GJR-GARCH(1,1) on raw XLK returns (no market factor) -> is asymmetry market-wide?
from arch import arch_model
raw = arch_model(ret.r_XLK, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="normal").fit(disp="off", cov_type="robust")
raw_tab = pd.DataFrame({"coef": raw.params, "robust se": raw.std_err, "p": raw.pvalues})
raw_tab.to_csv(os.path.join(RES, "t5_gjr_raw_xlk_returns.csv"))
log("\nBENCHMARK: GJR-GARCH(1,1) on raw XLK returns (constant mean, no market factor)")
log(raw_tab.round(4).to_string())

# Standardised residual diagnostics
z = fm["z"]
lb5 = pd.concat([lb_table(z, "z"), lb_table(z ** 2, "z^2")])
lb5.to_csv(os.path.join(RES, "t5_stdresid_ljungbox.csv"))
log("\nStandardised residuals z_t = e_t / sqrt(h_t):")
log(f"skew = {stats.skew(z):.3f}, excess kurtosis = {stats.kurtosis(z):.3f}")
log(lb5.round(4).to_string())
log(f"ARCH-LM(5) on z: p = {het_arch(z, nlags=5)[1]:.4f}")
log(f"corr(z_t^2, z_t-1) = {np.corrcoef(z[1:] ** 2, z[:-1])[0, 1]:.4f}")

# ----------------------------------------------------------------------------
# 6. PARAMETER RESTRICTIONS AND CONDITIONAL VARIANCE
# ----------------------------------------------------------------------------
section("6. PARAMETER CHECKS AND INFERENCE IN FINAL MODEL")
p, V, free, names = fm["p"], fm["V"], fm["free"], fm["names"]
idx = {n: i for i, n in enumerate(np.array(names)[free])}
P = dict(zip(names, p))


def wald_lin(R, r, label):
    """Wald test R*theta = r using robust covariance."""
    th = p[free]
    R = np.atleast_2d(R); d = R @ th - r
    W = float(d @ np.linalg.solve(R @ V @ R.T, d))
    return [label, W, 1 - stats.chi2.cdf(W, R.shape[0])]


def vec(**kw):
    v = np.zeros(free.sum())
    for k, val in kw.items():
        v[idx[k]] = val
    return v


a_, g_, b_, w_ = P["alpha_arch"], P["gamma (GJR)"], P["beta_garch"], P["omega"]
pers_simple = a_ + b_
pers_gjr = a_ + g_ / 2 + b_
uvar = w_ / (1 - pers_gjr)
hl = np.log(0.5) / np.log(pers_gjr)
g6 = [
    wald_lin(vec(alpha=1), 0, "H0: alpha = 0"),
    wald_lin(vec(beta_SP=1), 1, "H0: beta_SP = 1"),
    wald_lin(vec(beta_VIX=1, beta_SP=-1), 0, "H0: beta_VIX = beta_SP"),
    wald_lin(vec(beta_VIX=1), 0, "H0: beta_VIX = 0"),
    wald_lin(np.vstack([vec(d_good=1), vec(d_bad=1)]), np.zeros(2), "H0: d_good = d_bad = 0"),
    wald_lin(vec(d_good=1, d_bad=-1), 0, "H0: d_good = d_bad"),
    wald_lin(vec(**{"lambda (in-mean)": 1}), 0, "H0: lambda = 0 (no risk premium)"),
    wald_lin(vec(**{"gamma (GJR)": 1}), 0, "H0: gamma = 0 (symmetry)"),
    wald_lin(vec(**{"phi (news var)": 1}), 0, "H0: phi = 0 (news do not move variance)"),
    wald_lin(vec(alpha_arch=1, beta_garch=1), 1, "H0: alpha + beta = 1 (IGARCH)"),
    wald_lin(vec(alpha_arch=1, beta_garch=1, **{"gamma (GJR)": 0.5}), 1, "H0: alpha + gamma/2 + beta = 1"),
]
g6 = pd.DataFrame(g6, columns=["hypothesis", "Wald chi2", "p-value"]).set_index("hypothesis")
g6.to_csv(os.path.join(RES, "t6_garch_wald_tests.csv"))
log(g6.round(4).to_string())
log(f"\nomega = {w_:.4f} (> 0: {w_ > 0})")
log(f"alpha + beta = {pers_simple:.4f} (< 1: {pers_simple < 1})")
log(f"alpha + gamma/2 + beta = {pers_gjr:.4f} (< 1: {pers_gjr < 1})  -> half-life of a shock = {hl:.1f} days")
log(f"alpha + gamma >= 0: {a_ + g_ >= 0}")
log(f"Unconditional variance (non-news day) = {uvar:.4f} -> ann. vol = {np.sqrt(252 * uvar):.2f}%")
log(f"News-day variance shift phi = {P['phi (news var)']:.4f} "
    f"(= {100 * P['phi (news var)'] / uvar:.1f}% of unconditional daily variance)")
log(f"alpha annualised = {252 * P['alpha']:.2f}%;  risk premium at avg sigma: "
    f"lambda*mean(sigma) = {P['lambda (in-mean)'] * np.sqrt(fm['h']).mean():.4f}% per day")

h = pd.Series(fm["h"], index=ret.index)
pd.DataFrame({"h": h, "sigma_ann": np.sqrt(252 * h), "z": z}).to_csv(os.path.join(RES, "conditional_variance.csv"))
log("\nLargest conditional volatility dates (annualised %):")
log(np.sqrt(252 * h).nlargest(5).round(1).to_string())

fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
ax[0].plot(h.index, h, lw=0.7, color="C0")
ax[0].set_title("Conditional variance $h_t$ of the XLK market-adjusted shock — GJR-GARCH(1,1)-M with news in variance")
ax[0].set_ylabel("$h_t$ (%$^2$ per day)")
s = np.sqrt(252 * h)
ax[1].plot(s.index, s, lw=0.7, color="C0", label="XLK conditional (market-adjusted) vol, ann. %")
ax[1].plot(px.VIX.loc[s.index].index, px.VIX.loc[s.index], lw=0.7, color="C3", alpha=0.7, label="VIX")
nd = ret.index[ret.NEWS == 1]
ax[1].scatter(nd, s.loc[nd], s=8, color="k", zorder=3, label="Tech news days")
ax[1].set_ylabel("annualised %"); ax[1].legend(loc="upper left")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "f2_conditional_variance.png"), dpi=150); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ax[0].plot(ret.index, z, lw=0.4); ax[0].set_title("Standardised residuals $z_t$")
xs = np.linspace(-6, 6, 400)
ax[1].hist(z, bins=80, density=True, alpha=0.6); ax[1].plot(xs, stats.norm.pdf(xs), "r")
ax[1].set_title("Histogram of $z_t$ vs N(0,1)"); ax[1].set_xlim(-6, 6)
stats.probplot(z, dist="norm", plot=ax[2]); ax[2].set_title("QQ-plot of $z_t$")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "f3_std_residuals.png"), dpi=150); plt.close()

# News impact curve
ee = np.linspace(-5, 5, 200)
nic = w_ + (a_ + g_ * (ee < 0)) * ee ** 2 + b_ * uvar
plt.figure(figsize=(6, 4))
plt.plot(ee, nic, label="GJR-GARCH")
plt.plot(ee, w_ + (a_ + g_ / 2) * ee ** 2 + b_ * uvar, "--", label="symmetric (same persistence)")
plt.xlabel("$\\varepsilon_{t-1}$ (%)"); plt.ylabel("$h_t$"); plt.title("News impact curve — XLK")
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG, "f4_news_impact_curve.png"), dpi=150); plt.close()

with open(os.path.join(RES, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
print("\nDone. Tables in results/, figures in figures/.")
