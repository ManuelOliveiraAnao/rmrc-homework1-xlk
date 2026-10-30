"""Run only versions that include the AI-EU index."""
import estimacao as engine
from saida import save_full,save_holdout
MODELS=[('ai_no_retorno',True,False),('ai_na_variancia',False,True),('ai_em_ambos',True,True)]

def main():
    df,_,_=engine.load_data()
    split=int(.8*len(df))
    for folder,mean_extra,var_extra in MODELS:
        model=engine.fit(df,mean_extra,var_extra)
        save_full(model,df.index,folder)
        train=engine.fit(df.iloc[:split],mean_extra,var_extra,inference=False)
        save_holdout(train,df,split,folder,engine,mean_extra)
        print(folder,model['info']['AIC'],model['info']['BIC'])

if __name__=='__main__':main()
