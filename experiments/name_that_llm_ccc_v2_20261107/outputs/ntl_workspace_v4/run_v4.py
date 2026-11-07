import sys,json,hashlib,time,warnings
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from method import *


def metrics(y,p):
    return {'BA':score(y,p.argmax(1),np.arange(18)),'logloss':float(log_loss(y,p,labels=np.arange(18)))}


def main(path):
    start=time.time()
    d=load_data(path)
    dh=hashlib.sha256(path.read_bytes()).hexdigest()
    saved=np.load(ROOT/'work'/f'qwen25_7bi_nf4_l6l20_{dh[:12]}.npz')
    assert np.array_equal(saved['id'],d.id.to_numpy().astype(str))
    z=saved['features']; t=d.text.to_numpy(); g=d.prompt_id.to_numpy()
    classes=np.asarray(sorted(d.model.unique()))
    y=np.searchsorted(classes,d.model.to_numpy())
    styles=rich_style(t)
    folds=np.array_split(np.random.default_rng(20261110).permutation(sorted(np.unique(g))),5)
    oof={}; logs=[]; stress=[]
    for fold,vg in enumerate(folds):
        tr,va=np.flatnonzero(~np.isin(g,vg)),np.flatnonzero(np.isin(g,vg))
        ug=np.random.default_rng(20261210+fold).permutation(sorted(np.unique(g[tr])))
        fit=tr[~np.isin(g[tr],ug[:24])]; cal=tr[np.isin(g[tr],ug[:24])]
        print('Fold',fold+1,'inner crossfit',flush=True)
        ip,ic=crossfit_base(z,y,g,t,fit,20261240+fold)
        ib=fit_base(z,y,g,fit)
        cp,cc=predict_base(ib,z[cal],t[cal])
        proj=Projection().fit(z[fit])
        ir,is_,isup=proj.transform(z[fit],styles[fit],ic,ip)
        cr,cs,csup=proj.transform(z[cal],styles[cal],cc,cp)
        print('Fold',fold+1,'inner boosted fits',flush=True)
        iboost=fit_boosters(ir,is_,y[fit],ip,cal=(cr,cs,y[cal],cp))
        im=combine(iboost,ir,is_,ip,isup)
        cm=combine(iboost,cr,cs,cp,csup)
        print('Fold',fold+1,'rounds',iboost['rounds'],'outer crossfit',flush=True)
        op,oc=crossfit_base(z,y,g,t,tr,20261250+fold)
        ob=fit_base(z,y,g,tr)
        outerproj=Projection().fit(z[tr])
        xr,xs,support=outerproj.transform(z[tr],styles[tr],oc,op)
        oboost=fit_boosters(xr,xs,y[tr],op,rounds=iboost['rounds'])
        bundle={'base':ob,'projection':outerproj,'boosters':oboost,'classes':classes,
                'definition':'workspace_grouped_residual_correction_v4','data_sha256':dh}
        pred,vsup=predict(bundle,z[va],t[va])
        for k,p in pred.items():
            if k not in oof:oof[k]=np.zeros((len(d),18))
            oof[k][va]=p
        log={'fold':fold+1,'rounds':iboost['rounds'],
             'inner_training':{k:metrics(y[fit],p) for k,p in im.items()},
             'inner_calibration':{k:metrics(y[cal],p) for k,p in cm.items()},
             'outer':{k:metrics(y[va],p) for k,p in pred.items()},
             'correction_coverage':float(vsup.mean()),'offset_reconstruction_verified':True}
        logs.append(log)
        print(json.dumps({'fold':fold+1,'outer':log['outer'],'coverage':log['correction_coverage']}),flush=True)
        joblib.dump(bundle,ROOT/'work'/f'v4_fold_{fold+1}.joblib',compress=3)
        if fold==0:
            for name,seed,prob in [('random_half',20261202,np.full(len(va),.5)),
                  ('class_biased',20261203,np.where(y[va]<9,.25,.75))]:
                ix=va[np.random.default_rng(seed).random(len(va))<prob]
                pp,sp=predict(bundle,z[ix],t[ix])
                stress.append({'scenario':name,'rows':len(ix),'metrics':{k:metrics(y[ix],pp[k])
                           for k in ['BASE','TREE','NAIVE','NO_GUARD','WORKSPACE']},'coverage':float(sp.mean())})
    assert all(np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5) for p in oof.values())
    old=np.load(ROOT/'work'/'v3_qwen_oof.npz')
    base_match=bool(np.allclose(oof['BASE'],old['Q1'],atol=1e-8))
    assert base_match,'Baseline must exactly reproduce iteration3 Q1'
    pred={k:p.argmax(1) for k,p in oof.items()}
    boot=group_bootstrap(y,g,pred,np.arange(18),20261230,2000)
    result={'status':'COMPLETE','data_sha256':dh,'base_matches_iteration3':base_match,
          'metrics':{k:metrics(y,p) for k,p in oof.items()},
          'intervals':{k:{'BA_95pct':np.quantile(v,[.025,.975]).tolist(),
             'delta_BASE_95pct':np.quantile(v-boot['BASE'],[.025,.975]).tolist()} for k,v in boot.items()},
          'class_recall':{k:dict(zip(classes,recall_vector(y,p,np.arange(18)))) for k,p in pred.items()},
          'folds':logs,'stress':stress,'seconds':round(time.time()-start,2),
          'protocol_sha256':hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),
          'method_sha256':hashlib.sha256((HERE/'method.py').read_bytes()).hexdigest(),
          'scope':'custom integration, internal research-exposed prompt CV; global novelty unestablished'}
    (HERE/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(ROOT/'work'/'v4_oof_probabilities.npz',y=y,groups=g,**oof)
    print(json.dumps({k:result[k] for k in ['metrics','intervals','seconds']},indent=2),flush=True)


if __name__=='__main__':
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    with threadpool_limits(limits=4):main(Path(sys.argv[1]))
