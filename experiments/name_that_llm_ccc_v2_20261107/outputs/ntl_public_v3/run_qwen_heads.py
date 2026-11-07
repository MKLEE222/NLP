"""Fixed frozen-decoder heads and same-fold public-component comparison."""
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from public_stylegram import normalised_content_groups,eligible

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(HERE.parent/'ntl_ccc_v2'))
from run_v2 import load_data,score,group_bootstrap,recall_vector
from run_killtest import by_prompt_center


def main(data):
    start=time.time()
    dh=hashlib.sha256(data.read_bytes()).hexdigest()
    d=load_data(data)
    cache=ROOT/'work'/f'qwen25_7bi_nf4_l6l20_{dh[:12]}.npz'
    raw=np.load(cache,allow_pickle=False)
    assert np.array_equal(raw['id'],d.id.to_numpy().astype(str))
    features=raw['features']
    assert np.isfinite(features).all()
    y,g,texts=d.model.to_numpy(),d.prompt_id.to_numpy(),d.text.to_numpy()
    classes=np.asarray(sorted(d.model.unique()))
    old=np.load(ROOT/'work'/'v2_oof_predictions.npz',allow_pickle=True)
    assert np.array_equal(y,old['y'])
    q={k:np.zeros((len(d),len(classes))) for k in ['Q0','Q1']}
    folds=np.array_split(np.random.default_rng(20261110).permutation(sorted(np.unique(g))),5)
    logs=[]
    qcache=ROOT/'work'/'v3_qwen_oof.npz'
    qlog=HERE/'qwen_head_folds.json'
    cm=HERE/'qwen_head_cache_manifest.json'
    binding={'data_sha256':dh,'feature_sha256':json.loads(cache.with_suffix('.json').read_text(encoding='utf-8'))['feature_sha256'],
             'protocol_sha256':hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),
             'head_definition':'v3_20261107_C0025_plain_and_centred'}
    reuse=qcache.exists() and qlog.exists() and cm.exists() and json.loads(cm.read_text(encoding='utf-8'))==binding
    if reuse:
        saved=np.load(qcache)
        q={k:saved[k] for k in ['Q0','Q1']}
        logs=json.loads(qlog.read_text(encoding='utf-8'))
        print('Reusing fixed Qwen OOF heads',flush=True)
    for fold,vg in (enumerate(folds) if not reuse else []):
        tr,va=np.flatnonzero(~np.isin(g,vg)),np.flatnonzero(np.isin(g,vg))
        assert (old['fold'][va]==fold).all()
        print('Qwen head fold',fold+1,flush=True)
        scale=StandardScaler().fit(features[tr])
        model=LogisticRegression(C=.0025,max_iter=2000,class_weight='balanced',random_state=42)
        model.fit(scale.transform(features[tr]),y[tr])
        assert np.array_equal(model.classes_,classes)
        q['Q0'][va]=model.predict_proba(scale.transform(features[va]))
        centered=by_prompt_center(features[tr],g[tr])
        cscale=StandardScaler().fit(centered)
        twin=LogisticRegression(C=.0025,max_iter=2000,class_weight='balanced',random_state=42)
        twin.fit(cscale.transform(centered),y[tr])
        inferred=normalised_content_groups(texts[va])
        use=eligible(inferred)
        p=q['Q0'][va].copy()
        if use.any():
            p[use]=twin.predict_proba(cscale.transform(by_prompt_center(features[va][use],inferred[use])))
        q['Q1'][va]=p
        log={'fold':fold+1,'Q0_BA':score(y[va],classes[q['Q0'][va].argmax(1)],classes),
             'Q1_BA':score(y[va],classes[p.argmax(1)],classes),'centred_coverage':float(use.mean())}
        logs.append(log)
        print(json.dumps(log),flush=True)
    np.savez_compressed(ROOT/'work'/'v3_qwen_oof.npz',**q)
    qlog.write_text(json.dumps(logs,indent=2),encoding='utf-8')
    cm.write_text(json.dumps(binding,indent=2),encoding='utf-8')
    if not (ROOT/'work'/'v3_style_oof.npz').exists():
        print('Qwen heads complete; rerun after StyleGrams finishes to assemble results.',flush=True)
        return
    style=np.load(ROOT/'work'/'v3_style_oof.npz',allow_pickle=True)
    assert json.loads((HERE/'style_results.json').read_text(encoding='utf-8'))['data_sha256']==dh
    assert np.array_equal(y,style['y']) and np.array_equal(old['fold'],style['fold'])
    allp={k:style[k] for k in ['P0','P1','P2']}
    allp.update(q)
    allp['E']=.5*allp['P1']+.5*allp['Q0']
    pred={k:classes[v.argmax(1)] for k,v in allp.items()}
    pred['V2']=old['V2']
    boot=group_bootstrap(y,g,pred,classes,20261201,2000)
    result={'status':'COMPLETE_COMPONENT_SUPPLEMENT','full_public_0604_ensemble_reproduced':False,
        'data_sha256':dh,'OOF_BA':{k:score(y,v,classes) for k,v in pred.items()},'qwen_folds':logs,
        'intervals':{k:{'BA_95pct':np.quantile(v,[.025,.975]).tolist(),
             'delta_V2_95pct':np.quantile(v-boot['V2'],[.025,.975]).tolist(),
             'lower_adjusted':float(np.quantile(v-boot['V2'],.05/6))} for k,v in boot.items()},
        'class_recall':{k:dict(zip(classes,recall_vector(y,v,classes))) for k,v in pred.items()},
        'feature_provenance':json.loads(cache.with_suffix('.json').read_text(encoding='utf-8')),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'protocol_sha256':hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),
        'heads_reused':reuse,'seconds_this_invocation':round(time.time()-start,2),
        'scope':'fixed public components and resource adaptations; internal five-fold prompt CV, not public LB'}
    (HERE/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(ROOT/'work'/'v3_oof_predictions.npz',y=y,groups=g,**pred)
    print(json.dumps({k:result[k] for k in ['OOF_BA','intervals','heads_reused','seconds_this_invocation']},indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4): main(Path(sys.argv[1]))
