"""Run the no-index model on the exact common comparison dates."""
from pathlib import Path
import pandas as pd
import estimacao as engine
from saida import save_full,save_holdout
ROOT=Path(__file__).resolve().parents[1]

def main():
    df=pd.read_csv(ROOT/'dados/dataset.csv',index_col=0,parse_dates=True)
    # Fixed common dates, independent of any external index file.
    df=df.loc['2015-01-05':'2026-10-27']
    model=engine.fit(df)
    save_full(model,df.index,'amostra_comum')
    split=int(.8*len(df))
    train=engine.fit(df.iloc[:split],inference=False)
    save_holdout(train,df,split,'amostra_comum',engine)
    print('Sem indice: resultados/amostra_comum; n =',len(df))

if __name__=='__main__':main()
