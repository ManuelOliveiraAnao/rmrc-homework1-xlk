"""Save and plot one model; no cross-version dependencies."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def plot_index_series(ai, df):
    """Render the index and its lagged change with English labels."""
    out=ROOT/'figures';out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    axes[0].plot(ai.index,ai.daily_aieu_index,lw=.6)
    axes[0].set(title='AI-EU: coverage of AI, economics and uncertainty',
                ylabel='Index (2015-2021 average = 100)')
    axes[1].plot(df.index,df.change_lag1,lw=.5)
    axes[1].set(title='Model input: change lagged by one calendar day',
                ylabel='Change in log(1 + index)',xlabel='Date')
    fig.tight_layout();fig.savefig(out/'aieu_series.png',dpi=160);plt.close(fig)

def save_full(model, dates, folder):
    out=ROOT/'results'/folder;out.mkdir(parents=True,exist_ok=True)
    model['table'].to_csv(out/'coefficients.csv')
    pd.DataFrame([model['info']]).to_csv(out/'model_summary.csv',index=False)
    series=pd.DataFrame({'h':model['h'],'residual':model['e'],'loglik':model['ll']},index=dates)
    series.to_csv(out/'filtered_series.csv')
    plot_saved(folder)

def plot_saved(folder):
    data=pd.read_csv(ROOT/'results'/folder/'filtered_series.csv',index_col=0,parse_dates=True)
    out=ROOT/'figures'/folder;out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    axes[0].plot(data.index,np.sqrt(252*data.h),lw=.7)
    axes[0].set(title=folder.replace('_',' '),ylabel='Annualized residual volatility (%)')
    axes[1].plot(data.index,data.residual/np.sqrt(data.h),lw=.5)
    axes[1].set(ylabel='Standardized residual',xlabel='Date')
    fig.tight_layout();fig.savefig(out/'volatility_and_residuals.png',dpi=150);plt.close(fig)

def save_holdout(model, df, split, folder, engine, mean_extra=False):
    out=ROOT/'results'/folder
    arrays,_=engine.design(df,mean_extra,model['scale'])
    ll,h,e=engine.forecast_filter(model,arrays,split)
    model['table'].to_csv(out/'training_coefficients.csv')
    # Squared errors and variances are saved separately. The comparison script
    # constructs QLIKE with ONE common baseline-error proxy for all versions.
    pd.DataFrame({'h':h[split:],'error':e[split:],'squared_error':e[split:]**2,
                  'loglik':ll[split:]},index=df.index[split:]).to_csv(out/'holdout.csv')
    info=dict(n_train=split,n_test=len(df)-split,train_start=str(df.index[0].date()),
              train_end=str(df.index[split-1].date()),test_start=str(df.index[split].date()),
              test_end=str(df.index[-1].date()),MSE=float(np.mean(e[split:]**2)),
              negative_logscore=float(-np.mean(ll[split:])))
    (out/'holdout.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
