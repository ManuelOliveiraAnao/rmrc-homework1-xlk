"""Read the separate projects and rebuild the common comparison. No fitting."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent
MODELS=[('Original','01_without_aieu','common_sample'),
        ('+ AI-EU in mean','02_with_aieu','ai_in_mean'),
        ('+ AI-EU in variance','02_with_aieu','ai_in_variance'),
        ('+ AI-EU in both','02_with_aieu','ai_in_both')]

def main():
    summaries=[];evaluations=[];series=[]
    for label,version,folder in MODELS:
        source=PARENT/version/'results'/folder
        info=pd.read_csv(source/'model_summary.csv').iloc[0].to_dict()
        summaries.append(dict(model=label,**info))
        e=pd.read_csv(source/'holdout.csv',index_col=0)
        series.append(pd.read_csv(source/'filtered_series.csv',index_col=0,parse_dates=True))
        if not evaluations:base_proxy=e.error.to_numpy()**2;base_dates=e.index
        assert e.index.equals(base_dates),'Different evaluation dates'
        loss=base_proxy/e.h.to_numpy()-np.log(np.maximum(base_proxy,1e-12)/e.h.to_numpy())-1
        evaluations.append(dict(model=label,MSE=float(np.mean(e.error**2)),QLIKE=float(loss.mean()),n_test=len(e)))
    c=pd.DataFrame(summaries);h=pd.DataFrame(evaluations)
    assert c.n.nunique()==1
    for metric in ['AIC','BIC']:c['delta_'+metric]=c[metric]-c[metric].iloc[0]
    for metric in ['MSE','QLIKE']:h['delta_'+metric+'_pct']=100*(h[metric]/h[metric].iloc[0]-1)
    c.to_csv(ROOT/'results/model_comparison.csv',index=False)
    h.to_csv(ROOT/'results/holdout_comparison.csv',index=False)
    for filename,df,cols,titles in [
        ('comparison_aic_bic.png',c,['delta_AIC','delta_BIC'],['Difference in AIC','Difference in BIC']),
        ('comparison_holdout.png',h,['delta_MSE_pct','delta_QLIKE_pct'],['Change in MSE (%)','Change in QLIKE loss (%)'])]:
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        for ax,col,title in zip(axes,cols,titles):
            ax.barh(df.model,df[col]);ax.invert_yaxis();ax.axvline(0,color='grey',lw=.8)
            ax.set(title=title,xlabel='Relative to baseline; lower is better')
        fig.savefig(ROOT/'figures'/filename,dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for (label,_,_),s in zip(MODELS,series):ax.plot(s.index,np.sqrt(252*s.h),lw=.7,label=label)
    ax.set(ylabel='Annualized residual volatility (%)',xlabel='Date');ax.legend()
    fig.savefig(ROOT/'figures/comparison_volatility.png',dpi=160);plt.close(fig)
    print(c[['model','AIC','BIC']].to_string(index=False))
    print(h.to_string(index=False))

if __name__=='__main__':main()
