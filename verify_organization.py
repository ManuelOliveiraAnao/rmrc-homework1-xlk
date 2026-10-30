"""Verify the separate saved outputs and their comparison; no model fitting."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
MODELS=[('01_without_aieu','common_sample'),('02_with_aieu','ai_in_mean'),
        ('02_with_aieu','ai_in_variance'),('02_with_aieu','ai_in_both')]
comparison=pd.read_csv(ROOT/'03_comparison/results/model_comparison.csv')
holdout=pd.read_csv(ROOT/'03_comparison/results/holdout_comparison.csv')
for i,(version,model) in enumerate(MODELS):
    out=ROOT/version/'results'/model
    s=pd.read_csv(out/'filtered_series.csv',index_col=0)
    info=pd.read_csv(out/'model_summary.csv').iloc[0]
    coef=pd.read_csv(out/'coefficients.csv',index_col=0)
    ll=-.5*(np.log(2*np.pi)+np.log(s.h)+s.residual**2/s.h)
    assert np.allclose(ll,s.loglik)
    assert s.h.gt(0).all() and len(s)==2971
    assert np.isclose(info.LL,ll.sum())
    assert np.isclose(info.AIC,-2*ll.sum()+2*info.k)
    assert np.isclose(info.BIC,-2*ll.sum()+np.log(len(s))*info.k)
    assert np.isclose(info.AIC,comparison.iloc[i].AIC)
    assert np.isclose(info.BIC,comparison.iloc[i].BIC)
    if i==0:
        reference_dates=s.index
        assert not any('AI_EU' in str(name) for name in coef.index)
    else:
        assert s.index.equals(reference_dates)
        assert any('AI_EU' in str(name) for name in coef.index)
    h=pd.read_csv(out/'holdout.csv',index_col=0)
    if i==0:proxy=h.error.to_numpy()**2;test_dates=h.index
    assert h.index.equals(test_dates) and len(h)==595
    qlike=proxy/h.h.to_numpy()-np.log(np.maximum(proxy,1e-12)/h.h.to_numpy())-1
    assert np.isclose(qlike.mean(),holdout.iloc[i].QLIKE)
    assert np.isclose(np.mean(h.error**2),holdout.iloc[i].MSE)
assert not (ROOT/'01_without_aieu/data/All_Daily_AIEU_Data.csv').exists()
assert (ROOT/'02_with_aieu/data/All_Daily_AIEU_Data.csv').exists()
for p in ROOT.rglob('*.py'):
    compile(p.read_text(encoding='utf-8'),str(p),'exec')
print('OK: independent codes, separate outputs, aligned dates and consistent calculations.')
