"""Run only versions that include the AI-EU index."""
import estimation as engine
from output import save_full,save_holdout,plot_index_series
MODELS=[('ai_in_mean',True,False),('ai_in_variance',False,True),('ai_in_both',True,True)]

def main():
    df,ai,_=engine.load_data()
    plot_index_series(ai,df)
    split=int(.8*len(df))
    for folder,mean_extra,var_extra in MODELS:
        model=engine.fit(df,mean_extra,var_extra)
        save_full(model,df.index,folder)
        train=engine.fit(df.iloc[:split],mean_extra,var_extra,inference=False)
        save_holdout(train,df,split,folder,engine,mean_extra)
        print(folder,model['info']['AIC'],model['info']['BIC'])

if __name__=='__main__':main()
