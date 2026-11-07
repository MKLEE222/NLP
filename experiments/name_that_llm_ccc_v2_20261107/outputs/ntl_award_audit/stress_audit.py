import sys,json,time,warnings
from pathlib import Path
import numpy as np
import joblib
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(HERE.parent/'ntl_workspace_v4'))
from method import load_data,predict,predict_base,combine,rich_style,score,group_bootstrap


def singleton(bundle,z,texts):
    norms=np.linalg.norm(z,axis=1)
    ctx={'centered':np.zeros_like(z),'valid':np.zeros(len(z),bool),
         'diag':np.column_stack([np.full(len(z),np.log(2)),np.ones(len(z)),np.log1p(norms),np.zeros(len(z))]).astype(np.float32)}
    p,_=predict_base(bundle['base'],z,texts,ctx)
    rep,stack,support=bundle['projection'].transform(z,rich_style(texts),ctx,p)
    return combine(bundle['boosters'],rep,stack,p,support),support


def main():
    start=time.time()
    d=load_data(Path('D:/name-that-llm.zip'))
    raw=np.load(ROOT/'work'/'qwen25_7bi_nf4_l6l20_7c375db7dfee.npz')
    assert np.array_equal(raw['id'],d.id.to_numpy().astype(str))
    z=raw['features'];t=d.text.to_numpy();g=d.prompt_id.to_numpy()
    classes=np.asarray(sorted(d.model.unique()));y=np.searchsorted(classes,d.model.to_numpy())
    folds=np.array_split(np.random.default_rng(20261110).permutation(sorted(np.unique(g))),5)
    allrows={};logs=[]
    modes=['BASE','XGB_FULL','LGB_FULL','WORKSPACE']
    for fold,vg in enumerate(folds):
        va=np.flatnonzero(np.isin(g,vg))
        b=joblib.load(ROOT/'work'/f'v4_fold_{fold+1}.joblib')
        original,_=predict(b,z[va],t[va])
        scenarios=[('full',va),
           ('half',va[np.random.default_rng(20261300+fold).random(len(va))<.5]),
           ('thin20',va[np.random.default_rng(20261310+fold).random(len(va))<.2]),
           ('label_shift',va[np.random.default_rng(20261320+fold).random(len(va))<np.where(y[va]<9,.25,.75)]),
           ('claude_sparse',va[np.random.default_rng(20261330+fold).random(len(va))<np.where(y[va]<5,.1,.9)]),
           ('claude_dense',va[np.random.default_rng(20261340+fold).random(len(va))<np.where(y[va]<5,.9,.1)])]
        rng=np.random.default_rng(20261350+fold)
        scenarios.append(('one_per_prompt',np.array([rng.choice(np.flatnonzero(g==v)) for v in vg])))
        scenarios += [('chunks32',va),('chunks8',va),('singleton',va)]
        original_labels=original['XGB_FULL'].argmax(1)
        lookup={v:i for i,v in enumerate(va)}
        for name,ix in scenarios:
            if name=='singleton':pp,support=singleton(b,z[ix],t[ix])
            elif name.startswith('chunks'):
                size=int(name[6:]);pieces=[];sp=[]
                for k in range(0,len(ix),size):
                    block,ss=predict(b,z[ix[k:k+size]],t[ix[k:k+size]])
                    pieces.append(block);sp.append(ss)
                pp={m:np.concatenate([v[m] for v in pieces]) for m in modes}
                support=np.concatenate(sp)
            elif name=='full':pp,support=predict(b,z[ix],t[ix])
            else:pp,support=predict(b,z[ix],t[ix])
            sc,model=b['base']['raw']
            pred={m:pp[m].argmax(1) for m in modes}
            pred['RAW']=model.predict(sc.transform(z[ix]))
            orig=original_labels[[lookup[v] for v in ix]]
            record={'fold':fold+1,'scenario':name,'rows':len(ix),
                    'BA':{m:score(y[ix],v,np.arange(18)) for m,v in pred.items()},
                    'support_coverage':float(support.mean()),
                    'XGB_disagreement_with_full_batch':float((pred['XGB_FULL']!=orig).mean())}
            logs.append(record)
            if name not in allrows:allrows[name]={'ix':[],'pred':{m:[] for m in pred}}
            allrows[name]['ix'].extend(ix)
            for m,v in pred.items():allrows[name]['pred'][m].extend(v)
        # Verify vectorized singleton diagnostics equal literal one-row inference.
        vec,_=singleton(b,z[va[:2]],t[va[:2]])
        for j,i in enumerate(va[:2]):
            real,_=predict(b,z[i:i+1],t[i:i+1])
            assert np.allclose(vec['XGB_FULL'][j],real['XGB_FULL'][0],atol=1e-7)
        print('Audited fold',fold+1,flush=True)
    summary={}
    for name,values in allrows.items():
        ix=np.asarray(values['ix']);pred={m:np.asarray(v) for m,v in values['pred'].items()}
        boot=group_bootstrap(y[ix],g[ix],pred,np.arange(18),20261399,1000)
        summary[name]={'rows':len(ix),'class_counts':np.bincount(y[ix],minlength=18).tolist(),
           'BA':{m:score(y[ix],v,np.arange(18)) for m,v in pred.items()},
           'XGB_delta_BASE_95pct':np.quantile(boot['XGB_FULL']-boot['BASE'],[.025,.975]).tolist(),
           'mean_support_coverage':float(np.mean([v['support_coverage'] for v in logs if v['scenario']==name])),
           'mean_XGB_disagreement_with_full_batch':float(np.mean([v['XGB_disagreement_with_full_batch'] for v in logs if v['scenario']==name]))}
    result={'scope':'post-hoc audit of fixed models; no fitting or selection','summary':summary,
            'folds':logs,'singleton_vectorization_verified':True,'seconds':round(time.time()-start,2)}
    (HERE/'stress_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    with threadpool_limits(limits=4):main()
