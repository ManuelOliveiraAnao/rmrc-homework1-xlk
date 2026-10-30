"""Run the no-index model on the exact common comparison dates."""
from pathlib import Path
import pandas as pd
import estimation as engine
from output import save_full,save_holdout
ROOT=Path(__file__).resolve().parents[1]

def main():
    df=pd.read_csv(ROOT/'data/dataset.csv',index_col=0,parse_dates=True)
    # Fixed common dates, independent of any external index file.
    df=df.loc['2015-01-05':'2026-10-27']
    model=engine.fit(df)
    save_full(model,df.index,'common_sample')
    split=int(.8*len(df))
    train=engine.fit(df.iloc[:split],inference=False)
    save_holdout(train,df,split,'common_sample',engine)
    print('No index: results/common_sample; n =',len(df))

if __name__=='__main__':main()
