"""Public StyleGrams component and external-data ablations, fixed prompt folds."""
import sys,json,zipfile,hashlib,time,gc,re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from scipy.special import softmax
from sklearn.svm import LinearSVC
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from public_stylegram import StyleGrams,sparse_group_centre,normalised_content_groups,eligible

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(HERE.parent/'ntl_ccc_v2'))
from run_v2 import load_data,score,group_bootstrap


def normal(s):
    return re.sub(r'\s+',' ',str(s)).strip().lower()


def prepare_extra(path,d):
    with zipfile.ZipFile(path) as z:
        e=pd.read_csv(z.open('ntl_extra_passages_claude.csv'))
    before=len(e)
    e=e[e.model.isin(d.model.unique())].copy()
    e.text=e.text.astype(str).map(lambda s:re.sub(r'\s+',' ',re.sub(r'^\*\*(.*)\*\*$',r'\1',s.strip())).strip())
    e=e[e.text.str.split().str.len().between(30,300)]
    e=e.drop_duplicates(['model','text']).drop_duplicates(['prompt_id','model'])
    hashes=set(d.text.map(normal))
    overlap=e.text.map(normal).isin(hashes)
    removed=int(overlap.sum())
    e=e[~overlap].reset_index(drop=True)
    e.prompt_id='external_'+e.prompt_id.astype(str)
    info={'source':'seyitkaangunes/ntl-extra-api-passages','version':2,'license':'CC BY 4.0',
          'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'raw_rows':before,
          'used_rows':len(e),'exact_normalized_official_overlap_removed':removed,
          'class_counts':e.model.value_counts().to_dict(),
          'source_claim':'fresh prompt bank; semantic prompt disjointness is author-reported, not proven by ID difference'}
    return e,info


def fit_svc(x,y,weights=None):
    return LinearSVC(C=.5,random_state=42,max_iter=4000).fit(x,y,sample_weight=weights)


def main(data,extra):
    start=time.time()
    d=load_data(data)
    e,info=prepare_extra(extra,d)
    texts,y,g=d.text.to_numpy(),d.model.to_numpy(),d.prompt_id.to_numpy()
    classes=np.asarray(sorted(d.model.unique()))
    folds=np.array_split(np.random.default_rng(20261110).permutation(sorted(np.unique(g))),5)
    probs={k:np.zeros((len(d),len(classes))) for k in ['P0','P1','P2']}
    fold_id=np.full(len(d),-1)
    logs=[]
    for fold,vg in enumerate(folds):
        tr,va=np.flatnonzero(~np.isin(g,vg)),np.flatnonzero(np.isin(g,vg))
        print('Style fold',fold+1,'official only',flush=True)
        grams=StyleGrams().fit(texts[tr])
        a,b=grams.transform(texts[tr]),grams.transform(texts[va])
        clf=fit_svc(a,y[tr])
        assert list(clf.classes_)==list(classes)
        probs['P0'][va]=softmax(clf.decision_function(b),axis=1)
        del grams,a,b,clf
        gc.collect()
        print('Style fold',fold+1,'external weighted fit',flush=True)
        grams=StyleGrams().fit(list(texts[tr])+e.text.tolist())
        xt,xe,xv=grams.transform(texts[tr]),grams.transform(e.text),grams.transform(texts[va])
        counts=pd.Series(y[tr]).value_counts()
        ec=e.model.value_counts()
        wt=np.array([.5 if c in ec else 1. for c in y[tr]])
        we=np.array([.5*counts[c]/ec[c] for c in e.model])
        weights=np.r_[wt,we]
        yy=np.r_[y[tr],e.model]
        for c in classes:
            assert np.isclose(weights[yy==c].sum(),counts[c])
        clf=fit_svc(vstack([xt,xe]).tocsr(),yy,weights)
        p=softmax(clf.decision_function(xv),axis=1)
        probs['P1'][va]=p
        del clf
        print('Style fold',fold+1,'centred twin',flush=True)
        xc=sparse_group_centre(xt,g[tr])
        ex_c=sparse_group_centre(xe,e.prompt_id.to_numpy(),lo=2,hi=10**6)
        twin=fit_svc(vstack([xc,ex_c]).tocsr(),yy,weights)
        inferred=normalised_content_groups(texts[va])
        use=eligible(inferred)
        p2=p.copy()
        if use.any():
            xvc=sparse_group_centre(xv[use],inferred[use])
            p2[use]=softmax(twin.decision_function(xvc),axis=1)
            del xvc
        probs['P2'][va]=p2
        fold_id[va]=fold
        log={'fold':fold+1,'BA':{k:score(y[va],classes[v[va].argmax(1)],classes) for k,v in probs.items()},
             'group_ARI':float(adjusted_rand_score(g[va],inferred)),'centred_coverage':float(use.mean()),
             'external_weight_class_totals_equal_official':True}
        logs.append(log)
        print(json.dumps(log),flush=True)
        del grams,xt,xe,xv,xc,ex_c,twin
        gc.collect()
    pred={k:classes[v.argmax(1)] for k,v in probs.items()}
    old=np.load(ROOT/'work'/'v2_oof_predictions.npz',allow_pickle=True)
    assert np.array_equal(y,old['y']) and np.array_equal(fold_id,old['fold'])
    pred['V2']=old['V2']
    boot=group_bootstrap(y,g,pred,classes,20261201,2000)
    result={'status':'COMPLETE','data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
            'source':'saddamaditya/name-that-llm-two-stage-lora-hungarian v1 embedded StyleGrams',
            'external':info,'folds':logs,'OOF_BA':{k:score(y,v,classes) for k,v in pred.items()},
            'intervals':{k:{'BA_95pct':np.quantile(v,[.025,.975]).tolist(),
                'delta_V2_95pct':np.quantile(v-boot['V2'],[.025,.975]).tolist(),
                'lower_adjusted':float(np.quantile(v-boot['V2'],.05/6))} for k,v in boot.items()},
            'seconds':round(time.time()-start,2)}
    (HERE/'style_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(ROOT/'work'/'v3_style_oof.npz',y=y,groups=g,fold=fold_id,classes=classes,**probs)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main(Path(sys.argv[1]),Path(sys.argv[2]))
