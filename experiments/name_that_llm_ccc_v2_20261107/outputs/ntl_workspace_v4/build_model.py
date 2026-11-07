"""Fit the predeclared workspace integration on all official training data."""
import sys,json,hashlib,warnings
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from method import *


def main(path):
    d=load_data(path)
    dh=hashlib.sha256(path.read_bytes()).hexdigest()
    cache=ROOT/'work'/f'qwen25_7bi_nf4_l6l20_{dh[:12]}.npz'
    feature=np.load(cache)
    assert np.array_equal(feature['id'],d.id.to_numpy().astype(str))
    z=feature['features']; t=d.text.to_numpy(); g=d.prompt_id.to_numpy()
    classes=np.asarray(sorted(d.model.unique())); y=np.searchsorted(classes,d.model.to_numpy())
    evidence=json.loads((HERE/'results.json').read_text(encoding='utf-8'))
    rounds={k:int(np.median([x['rounds'][k] for x in evidence['folds']])) for k in ['plain','xgb','lgb']}
    ix=np.arange(len(d))
    print('Full-data prompt crossfit for correction targets',flush=True)
    op,oc=crossfit_base(z,y,g,t,ix,20261260)
    proj=Projection().fit(z)
    rep,stack,support=proj.transform(z,rich_style(t),oc,op)
    boosters=fit_boosters(rep,stack,y,op,rounds=rounds)
    base=fit_base(z,y,g,ix)
    bundle={'base':base,'projection':proj,'boosters':boosters,'classes':classes,
            'definition':'workspace_grouped_residual_correction_v4','data_sha256':dh,
            'encoder_provenance':json.loads(cache.with_suffix('.json').read_text(encoding='utf-8'))}
    out=HERE/'workspace_residual_model.joblib'
    joblib.dump(bundle,out,compress=3)
    reloaded=joblib.load(out)
    parity_ix=np.flatnonzero(np.isin(g,np.unique(g)[:5]))
    a,sa=predict(bundle,z[parity_ix],t[parity_ix])
    b,sb=predict(reloaded,z[parity_ix],t[parity_ix])
    assert np.array_equal(sa,sb) and all(np.allclose(a[k],b[k],atol=1e-12) for k in a)
    info={'artifact':out.name,'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
          'rounds':rounds,'trained_rows':len(d),'reload_probability_parity':True,
          'parity_rows':len(parity_ix),'parity_corrected_rows':int(sa.sum()),
          'requires_backbone':'same pinned Qwen2.5-7B-Instruct/NF420-layer encoder',
          'scope':'full-data candidate artifact; outcome justification remains in the frozen CV report'}
    (HERE/'model_manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    with threadpool_limits(limits=4):main(Path(sys.argv[1]))
