"""Supplementary OLS and likelihood-ratio reference tables; run after compare.py."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import linear_reset

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '03_comparison/results'
HAC = {'maxlags': 10}


def main():
    d = pd.read_csv(ROOT / '02_with_aieu/results/data_quality/dataset_aieu.csv', index_col=0, parse_dates=True)
    X = sm.add_constant(d[['r_SP', 'dVIX', 'GOOD', 'BAD']])
    u = (d.change_lag1 - d.change_lag1.mean()) / d.change_lag1.std(ddof=0)
    rows = []
    for name, XX in [('OLS original', X), ('OLS + AI-EU', X.assign(AI_EU=u))]:
        m = sm.OLS(d.r_XLK, XX).fit()
        h = sm.OLS(d.r_XLK, XX).fit(cov_type='HAC', cov_kwds=HAC)
        reset = linear_reset(m, power=3, use_f=True, cov_type='HAC', cov_kwds=HAC).pvalue
        rows.append(dict(model=name, R2=m.rsquared, adj_R2=m.rsquared_adj, AIC=m.aic, BIC=m.bic,
                         RESET_HAC_p=reset, AI_coef=h.params.get('AI_EU', np.nan),
                         AI_HAC_p=h.pvalues.get('AI_EU', np.nan)))
    pd.DataFrame(rows).to_csv(OUT / 'ols_comparison.csv', index=False)

    cmp_ = pd.read_csv(OUT / 'model_comparison.csv')
    base = cmp_.iloc[0]
    caveat = ('Gaussian reference only; nonnegative variance coefficient on boundary; '
              'robust Wald preferred for mean.')
    lr = []
    for _, r in cmp_.iloc[1:].iterrows():
        stat, df = 2 * (r.LL - base.LL), int(r.k - base.k)
        lr.append(dict(model=r.model, LR=stat, df=df, p_reference=1 - stats.chi2.cdf(stat, df), caveat=caveat))
    pd.DataFrame(lr).to_csv(OUT / 'likelihood_ratio_reference.csv', index=False)

    orig = pd.read_csv(ROOT / '01_without_aieu/results/original_sample/t5_model_comparison.csv', index_col=0)
    vfile = OUT / 'validation.json'
    v = json.loads(vfile.read_text(encoding='utf-8'))
    v['original_full_sample_replication_LL'] = float(orig.loc['GJR-GARCH(1,1)-M-X', 'LL'])
    vfile.write_text(json.dumps(v, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(pd.DataFrame(lr).drop(columns='caveat').round(4).to_string(index=False))


if __name__ == '__main__':
    main()
