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

@njit(cache=True)
def recursion(p, y, x, news, magnitude):
    """p = mean coefficients; lambda, omega, arch, gamma, garch, phi."""
    k = x.shape[1]
    lam, omega, a, gamma, beta, phi = p[k:k+6]
    n = len(y)
    mu = x @ p[:k]
    h = np.empty(n)
    e = np.empty(n)
    ll = np.empty(n)
    h[0] = np.var(y - mu)
    for t in range(n):
        if t:
            ep = e[t-1]
            h[t] = omega + (a + gamma * (ep < 0)) * ep**2 + beta*h[t-1] + phi*news[t]
        if h[t] <= 0 or not np.isfinite(h[t]):
            return np.full(n, -1e10), h, e
        e[t] = y[t] - mu[t] - lam*np.sqrt(h[t])
        ll[t] = -.5*(np.log(2*np.pi) + np.log(h[t]) + e[t]**2/h[t])
    return ll, h, e

def design(df, include_mean=False, scale=None):
    if include_mean:
        raise ValueError('This version estimates only the baseline mean.')
    x=np.ascontiguousarray(np.column_stack([np.ones(len(df)),df.r_SP,df.dVIX,df.GOOD,df.BAD]))
    return (df.r_XLK.to_numpy(),x,df.NEWS.to_numpy(dtype=float),np.zeros(len(df))), {}

def fit(df, include_mean=False, include_var=False, scale=None, extra_start=None, inference=True):
    if include_mean or include_var:
        raise ValueError('This version estimates only the baseline model.')
    arrays, scale = design(df, include_mean, scale)
    y, x, news, magnitude = arrays
    n, k = x.shape
    b0 = np.linalg.lstsq(x, y, rcond=None)[0]
    names = ['alpha', 'beta_SP', 'beta_VIX', 'd_good', 'd_bad']
    names += ['lambda', 'omega', 'alpha_arch', 'gamma', 'beta_garch', 'phi_news']
    template = np.r_[b0, -.045, .0057, .0797, -.0135, .9082, .0258]
    active = np.ones(len(template), dtype=bool)
    bounds = [(None,None)]*k + [(-2,2), (1e-8,5), (0,1), (-1,1), (0,.9999), (0,50)]
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
                optimizer_success=bool(r.success), starts_converged=len(solutions), hessian_condition=condition)
    return dict(p=p, names=names, table=table, info=info, h=h, e=e, ll=ll, scale=scale,
                include_mean=include_mean, include_var=include_var, params=dict(zip(names,p)))

def forecast_filter(model,arrays,split):
    # Same recursion as estimation, with h0 explicitly based on training only.
    p=model['p'];y,x,news,mag=arrays;k=x.shape[1]
    h0=np.var(y[:split]-x[:split]@p[:k])
    return recursion_holdout(p,y,x,news,mag,h0)

@njit(cache=True)
def recursion_holdout(p,y,x,news,mag,h0):
    k=x.shape[1];lam,w,a,g,b,phi=p[k:k+6]
    h=np.empty(len(y));e=np.empty(len(y));ll=np.empty(len(y));h[0]=h0
    for t in range(len(y)):
        if t:h[t]=w+(a+g*(e[t-1]<0))*e[t-1]**2+b*h[t-1]+phi*news[t]
        e[t]=y[t]-x[t]@p[:k]-lam*np.sqrt(h[t])
        ll[t]=-.5*(np.log(2*np.pi)+np.log(h[t])+e[t]**2/h[t])
    return ll,h,e
