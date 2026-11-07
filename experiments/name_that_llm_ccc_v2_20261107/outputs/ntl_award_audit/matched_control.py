import sys,json,time,warnings
from pathlib import Path
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(HERE.parent/'ntl_workspace_v4'))
from method import load_data,rich_style,context,score,group_bootstrap


def main():
    start=time.time()
    d=load_data(Path('D:/name-that-llm.zip'))
    z=np.load(ROOT/'work'/'qwen25_7bi_nf4_l6l20_7c375db7dfee.npz')['features']
    t,g=d.text.to_numpy(),d.prompt_id.to_numpy()
    classes=np.asarray(sorted(d.model.unique()));y=np.searchsorted(classes,d.model.to_numpy())
    styles=rich_style(t)
    original=np.load(ROOT/'work'/'v4_oof_probabilities.npz',allow_pickle=True)
    folds=np.array_split(np.random.default_rng(20261110).permutation(sorted(np.unique(g))),5)
    pred=np.full(len(d),-1);logs=[]
    for fold,vg in enumerate(folds):
        tr,va=np.flatnonzero(~np.isin(g,vg)),np.flatnonzero(np.isin(g,vg))
        bundle=joblib.load(ROOT/'work'/f'v4_fold_{fold+1}.joblib')
        ctx={'centered':np.zeros_like(z[tr]),'diag':np.zeros((len(tr),4),np.float32),'valid':np.zeros(len(tr),bool)}
        parts=np.array_split(np.random.default_rng(20261250+fold).permutation(sorted(np.unique(g[tr]))),3)
        for hold in parts:
            which=np.isin(g[tr],hold)
            c=context(z[tr][which],t[tr][which])
            for k in ctx:ctx[k][which]=c[k]
        rep,_,_=bundle['projection'].transform(z[tr],styles[tr],ctx,np.full((len(tr),18),1/18))
        ev,_,_=bundle['projection'].transform(z[va],styles[va],context(z[va],t[va]),np.full((len(va),18),1/18))
        check=bundle['boosters']['plain'].predict_proba(ev,iteration_range=(0,bundle['boosters']['rounds']['plain']))
        assert np.allclose(check,original['TREE'][va],atol=1e-7),'Feature reconstruction mismatch'
        sc=StandardScaler().fit(rep)
        clf=LogisticRegression(C=1,max_iter=2000,class_weight='balanced',random_state=42).fit(sc.transform(rep),y[tr])
        pred[va]=clf.predict(sc.transform(ev))
        logs.append({'fold':fold+1,'compact_LR_BA':score(y[va],pred[va],np.arange(18)),
                     'feature_dimensions':rep.shape[1],'tree_prediction_reconstructed':True})
    pp={'COMPACT_LR':pred,'BASE':original['BASE'].argmax(1),'TREE':original['TREE'].argmax(1),
        'XGB_FULL':original['XGB_FULL'].argmax(1)}
    boot=group_bootstrap(y,g,pp,np.arange(18),20261400,2000)
    result={'scope':'post-hoc fixed matched-feature linear control; no C tuning',
        'BA':{k:score(y,v,np.arange(18)) for k,v in pp.items()},'folds':logs,
        'LR_minus_TREE_95pct':np.quantile(boot['COMPACT_LR']-boot['TREE'],[.025,.975]).tolist(),
        'XGB_residual_minus_LR_95pct':np.quantile(boot['XGB_FULL']-boot['COMPACT_LR'],[.025,.975]).tolist(),
        'seconds':round(time.time()-start,2)}
    (HERE/'matched_control_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    with threadpool_limits(limits=4):main()
