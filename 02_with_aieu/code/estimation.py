"""Local estimation engine. All paths belong to this version only."""
from pathlib import Path
import hashlib, json, warnings
import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize
from scipy.stats import norm
from statsmodels.tools.numdiff import approx_hess, approx_fprime
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'data_quality'
OUT.mkdir(parents=True,exist_ok=True)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def load_data():
    ret = pd.read_csv(ROOT / 'data/dataset.csv', index_col=0, parse_dates=True)
    assert ret.index.is_unique and ret.index.is_monotonic_increasing
    raw = pd.read_csv(ROOT / 'data/All_Daily_AIEU_Data.csv')
    raw['date'] = pd.to_datetime(raw[['year', 'month', 'day']])
    duplicate_count = int(raw.date.duplicated().sum())
    for _, group in raw.groupby('date'):
        if len(group) > 1 and len(group.drop_duplicates()) != 1:
            raise ValueError('Conflicting AI-EU values for one date')
    ai = raw.drop_duplicates('date').set_index('date').sort_index()
    assert ai.index.equals(pd.date_range(ai.index.min(), ai.index.max(), name='date'))
    assert ai.daily_aieu_index.notna().all() and (ai.daily_aieu_index >= 0).all()
    ai['log_aieu'] = np.log1p(ai.daily_aieu_index)
    ai['change'] = ai.log_aieu.diff()
    # Shift on the full CALENDAR before joining to trading dates. Monday uses
    # Sunday's change, not Friday's. Never fill absent data or turn missing into zero.
    ai['change_lag1'] = ai.change.shift(1)
    ai['magnitude_lag1'] = ai.change.abs().shift(1)
    ai['level_lag1'] = ai.daily_aieu_index.shift(1)
    joined = ret.join(ai[['daily_aieu_index', 'level_lag1', 'change_lag1', 'magnitude_lag1']])
    joined = joined.dropna(subset=['change_lag1', 'magnitude_lag1'])
    assert len(joined) <= len(ret) and joined.index.is_unique
    # Confirm returns independently against supplied prices, avoiding silently
    # carrying a column-name or percentage-scale error into the comparison.
    prices = pd.read_csv(ROOT / 'data/prices_raw.csv', index_col=0, parse_dates=True).dropna()
    for target, source in [('r_XLK', 'XLK'), ('r_SP', '^GSPC'), ('dVIX', '^VIX')]:
        expected = 100 * np.log(prices[source]).diff()
        assert np.allclose(joined[target], expected.loc[joined.index], atol=1e-10)
    rows = []
    for name, values in [('AI-EU level', ai.daily_aieu_index), ('log(1+AI-EU)', ai.log_aieu),
                         ('delta log(1+AI-EU)', ai.change.dropna()),
                         ('abs(delta log(1+AI-EU))', ai.change.dropna().abs())]:
        rows.append(dict(series=name, ADF_p=adfuller(values, autolag='AIC')[1],
                         KPSS_p=kpss(values, regression='c', nlags='auto')[1]))
    pd.DataFrame(rows).to_csv(OUT / 'stationarity.csv', index=False)
    joined.to_csv(OUT / 'dataset_aieu.csv')
    metadata = dict(source='https://www.policyuncertainty.com/aieu_daily.html',
                    download='https://www.policyuncertainty.com/media/All_Daily_AIEU_Data.csv',
                    sha256=hashlib.sha256((ROOT/'data/All_Daily_AIEU_Data.csv').read_bytes()).hexdigest(),
                    source_first=str(ai.index.min().date()), source_last=str(ai.index.max().date()),
                    identical_duplicates_removed=duplicate_count,
                    calendar_gaps=0, original_n=len(ret), comparison_n=len(joined),
                    comparison_first=str(joined.index.min().date()), comparison_last=str(joined.index.max().date()),
                    dropped_trading_dates=[str(d.date()) for d in ret.index.difference(joined.index)],
                    alignment='Previous calendar day change; lag applied before trading-day join',
                    revisions='Source revises last 30 days. This is a revised-data pseudo holdout, not a real-time backtest.',
                    evaluation='Conditional holdout uses same-day observed market and VIX. Original earnings labels use full-sample medians. Not an ex-ante trading forecast.',
                    magnitude_stationarity='Absolute changes reject level stationarity under KPSS; variance extension is exploratory.',
                    inherited_limits='Original earnings labels and dates retained, including full-sample medians and original event mapping. No causal claim.',
                    license='CC BY 4.0; Economic Policy Uncertainty / policyuncertainty.com')
    (OUT / 'source_and_checks.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return joined, ai, metadata

@njit(cache=True)
def recursion(p, y, x, news, magnitude):
    """p = mean coefficients; lambda, omega, arch, gamma, garch, phi, psi."""
    k = x.shape[1]
    lam, omega, a, gamma, beta, phi, psi = p[k:k+7]
    n = len(y)
    mu = x @ p[:k]
    h = np.empty(n)
    e = np.empty(n)
    ll = np.empty(n)
    h[0] = np.var(y - mu)
    for t in range(n):
        if t:
            ep = e[t-1]
            h[t] = omega + (a + gamma * (ep < 0)) * ep**2 + beta*h[t-1] + phi*news[t] + psi*magnitude[t]
        if h[t] <= 0 or not np.isfinite(h[t]):
            return np.full(n, -1e10), h, e
        e[t] = y[t] - mu[t] - lam*np.sqrt(h[t])
        ll[t] = -.5*(np.log(2*np.pi) + np.log(h[t]) + e[t]**2/h[t])
    return ll, h, e

def design(df, include_mean, scale=None):
    if scale is None:
        scale = dict(center=float(df.change_lag1.mean()), sd=float(df.change_lag1.std(ddof=0)),
                     mag_sd=float(df.magnitude_lag1.std(ddof=0)))
    x = np.column_stack([np.ones(len(df)), df.r_SP, df.dVIX, df.GOOD, df.BAD])
    if include_mean:
        x = np.column_stack([x, (df.change_lag1.to_numpy()-scale['center'])/scale['sd']])
    return (df.r_XLK.to_numpy(), np.ascontiguousarray(x), df.NEWS.to_numpy(dtype=float),
            df.magnitude_lag1.to_numpy()/scale['mag_sd']), scale

def fit(df, include_mean=False, include_var=False, scale=None, extra_start=None, inference=True):
    arrays, scale = design(df, include_mean, scale)
    y, x, news, magnitude = arrays
    n, k = x.shape
    b0 = np.linalg.lstsq(x, y, rcond=None)[0]
    names = ['alpha', 'beta_SP', 'beta_VIX', 'd_good', 'd_bad']
    if include_mean:
        names += ['AI_EU_change_mean']
    names += ['lambda', 'omega', 'alpha_arch', 'gamma', 'beta_garch', 'phi_news', 'AI_EU_magnitude_variance']
    # Zero additional variance coefficient is an exact nesting of the baseline.
    template = np.r_[b0, -.045, .0057, .0797, -.0135, .9082, .0258, 0.]
    active = np.ones(len(template), dtype=bool)
    if not include_var:
        active[-1] = False
    bounds = [(None,None)]*k + [(-2,2), (1e-8,5), (0,1), (-1,1), (0,.9999), (0,50), (0,50)]
    bounds = [b for b, use in zip(bounds, active) if use]

    def expand(q):
        p = template.copy()
        p[active] = q
        return p

    def objective(q):
        return -recursion(expand(q), *arrays)[0].mean()

    def restriction(q):
        p = expand(q)
        a, g, beta = p[k+2:k+5]
        return np.array([a+g, .999999-a-.5*g-beta])

    candidates = [template[active]]
    # Several valid starting persistence values reduce optimizer-start sensitivity.
    for a, g, beta in [(.05,.05,.8), (.12,-.04,.8), (.1,0,.6)]:
        p = template.copy(); p[k+2:k+5] = [a,g,beta]
        if include_var: p[-1] = .01
        candidates.append(p[active])
    if extra_start:
        p = template.copy()
        for i, name in enumerate(names):
            if name in extra_start: p[i] = extra_start[name]
        candidates.append(p[active])
    solutions = []
    for start in candidates:
        r = minimize(objective, start, method='SLSQP', bounds=bounds,
                     constraints=[dict(type='ineq', fun=restriction)],
                     options=dict(maxiter=1800, ftol=1e-11))
        if r.success and np.min(restriction(r.x)) >= -1e-7 and np.isfinite(r.fun):
            solutions.append(r)
    if not solutions:
        raise RuntimeError('No converged valid optimum')
    r = min(solutions, key=lambda z:z.fun)
    p = expand(r.x)
    ll,h,e = recursion(p,*arrays)
    assert np.all(h > 0)
    table = pd.DataFrame({'coef':p[active]}, index=np.array(names)[active])
    condition = np.nan
    if inference:
        llvec = lambda q: recursion(expand(q),*arrays)[0]
        H = -approx_hess(r.x, lambda q:llvec(q).sum())
        G = approx_fprime(r.x, llvec)
        condition = float(np.linalg.cond(H))
        Hinv = np.linalg.inv(H)
        covariance = Hinv @ (G.T@G) @ Hinv
        assert np.all(np.diag(covariance) > 0)
        table['robust_se'] = np.sqrt(np.diag(covariance))
        table['z'] = table.coef / table.robust_se
        table['p'] = 2*norm.sf(np.abs(table.z))
        table['ci95_low'] = table.coef-1.96*table.robust_se
        table['ci95_high'] = table.coef+1.96*table.robust_se
    z = e / np.sqrt(h)
    info = dict(n=n, k=int(active.sum()), LL=float(ll.sum()), AIC=-2*ll.sum()+2*active.sum(),
                BIC=-2*ll.sum()+np.log(n)*active.sum(),
                LB_z_10_p=acorr_ljungbox(z,lags=[10]).lb_pvalue.iloc[0],
                LB_z2_5_p=acorr_ljungbox(z*z,lags=[5]).lb_pvalue.iloc[0],
                ARCH_LM5_p=het_arch(z,nlags=5)[1],
                persistence=p[k+2]+.5*p[k+3]+p[k+4],
                optimizer_success=bool(r.success), starts_converged=len(solutions), hessian_condition=condition,
                variance_ai_at_boundary=bool(include_var and p[-1] < 1e-7))
    return dict(p=p, names=names, table=table, info=info, h=h, e=e, ll=ll, scale=scale,
                include_mean=include_mean, include_var=include_var, params=dict(zip(names,p)))

def forecast_filter(model,arrays,split):
    # Same recursion as estimation, with h0 explicitly based on training only.
    p=model['p'];y,x,news,mag=arrays;k=x.shape[1]
    h0=np.var(y[:split]-x[:split]@p[:k])
    return recursion_holdout(p,y,x,news,mag,h0)

@njit(cache=True)
def recursion_holdout(p,y,x,news,mag,h0):
    k=x.shape[1];lam,w,a,g,b,phi,psi=p[k:k+7]
    h=np.empty(len(y));e=np.empty(len(y));ll=np.empty(len(y));h[0]=h0
    for t in range(len(y)):
        if t:h[t]=w+(a+g*(e[t-1]<0))*e[t-1]**2+b*h[t-1]+phi*news[t]+psi*mag[t]
        e[t]=y[t]-x[t]@p[:k]-lam*np.sqrt(h[t])
        ll[t]=-.5*(np.log(2*np.pi)+np.log(h[t])+e[t]**2/h[t])
    return ll,h,e
