import sys,json,time
from pathlib import Path
import numpy as np
from scipy.sparse import vstack
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from run_style import HERE,ROOT,load_data,prepare_extra,StyleGrams,fit_svc,sparse_group_centre,normalised_content_groups,eligible,score
from run_killtest import by_prompt_center


def main():
    start=time.time()
    d=load_data(Path('D:/name-that-llm.zip'))
    e,_=prepare_extra(ROOT/'work'/'public_refs'/'extra_v2.zip',d)
    y,g,t=d.model.to_numpy(),d.prompt_id.to_numpy(),d.text.to_numpy()
    classes=np.asarray(sorted(d.model.unique()))
    vg=np.array_split(np.random.default_rng(20261110).permutation(sorted(np.unique(g))),5)[0]
    tr,va=np.flatnonzero(~np.isin(g,vg)),np.flatnonzero(np.isin(g,vg))
    grams=StyleGrams().fit(list(t[tr])+e.text.tolist())
    xt,xe=grams.transform(t[tr]),grams.transform(e.text)
    counts={c:int((y[tr]==c).sum()) for c in classes}
    ec=e.model.value_counts()
    wt=np.array([.5 if c in ec else 1. for c in y[tr]])
    we=np.array([.5*counts[c]/ec[c] for c in e.model])
    yy=np.r_[y[tr],e.model]
    plain=fit_svc(vstack([xt,xe]).tocsr(),yy,np.r_[wt,we])
    twin=fit_svc(vstack([sparse_group_centre(xt,g[tr]),
         sparse_group_centre(xe,e.prompt_id.to_numpy(),lo=2,hi=10**6)]).tocsr(),yy,np.r_[wt,we])
    import hashlib
    dh=hashlib.sha256(Path('D:/name-that-llm.zip').read_bytes()).hexdigest()
    z=np.load(ROOT/'work'/f'qwen25_7bi_nf4_l6l20_{dh[:12]}.npz')['features']
    sc=StandardScaler().fit(z[tr])
    qp=LogisticRegression(C=.0025,max_iter=2000,class_weight='balanced',random_state=42).fit(sc.transform(z[tr]),y[tr])
    zc=by_prompt_center(z[tr],g[tr])
    cs=StandardScaler().fit(zc)
    qc=LogisticRegression(C=.0025,max_iter=2000,class_weight='balanced',random_state=42).fit(cs.transform(zc),y[tr])
    old=np.load(ROOT/'work'/'v2_oof_predictions.npz',allow_pickle=True)
    rows=[]
    for name,seed,p in [('random_half',20261202,np.full(len(va),.5)),
                        ('class_biased',20261203,np.where(np.isin(y[va],classes[:9]),.25,.75))]:
        ix=va[np.random.default_rng(seed).random(len(va))<p]
        inferred=normalised_content_groups(t[ix])
        use=eligible(inferred)
        x=grams.transform(t[ix])
        pp=softmax(plain.decision_function(x),axis=1)
        pc=pp.copy()
        if use.any():pc[use]=softmax(twin.decision_function(sparse_group_centre(x[use],inferred[use])),axis=1)
        q0=qp.predict_proba(sc.transform(z[ix]))
        q1=q0.copy()
        if use.any():q1[use]=qc.predict_proba(cs.transform(by_prompt_center(z[ix][use],inferred[use])))
        preds={k:classes[v.argmax(1)] for k,v in {'P1':pp,'P2':pc,'Q0':q0,'Q1':q1,'E':.5*pp+.5*q0}.items()}
        preds['V2']=old['V2'][ix]
        row={'scenario':name,'rows':len(ix),'BA':{k:score(y[ix],v,classes) for k,v in preds.items()},
             'group_ARI':float(adjusted_rand_score(g[ix],inferred)),'centred_coverage':float(use.mean()),
             'class_counts':{c:int((y[ix]==c).sum()) for c in classes}}
        rows.append(row)
        print(json.dumps(row),flush=True)
    result={'scope':'predeclared single-fold incomplete-batch diagnostic','fold':1,'scenarios':rows,
            'seconds':round(time.time()-start,2)}
    (HERE/'stress_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
