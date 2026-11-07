"""Export the fixed plain/centred Qwen heads, using the existing frozen feature cache."""
import sys,json,hashlib
from pathlib import Path
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from run_qwen_heads import HERE,ROOT,load_data,by_prompt_center


def main(path):
    d=load_data(path)
    dh=hashlib.sha256(path.read_bytes()).hexdigest()
    f=ROOT/'work'/f'qwen25_7bi_nf4_l6l20_{dh[:12]}.npz'
    z=np.load(f)['features']
    bundle={'classes':np.asarray(sorted(d.model.unique())),'data_sha256':dh,
            'feature_provenance':json.loads(f.with_suffix('.json').read_text(encoding='utf-8'))}
    for name,x in [('plain',z),('centred',by_prompt_center(z,d.prompt_id.to_numpy()))]:
        sc=StandardScaler().fit(x)
        clf=LogisticRegression(C=.0025,max_iter=2000,class_weight='balanced',random_state=42).fit(sc.transform(x),d.model)
        bundle[name]={'scaler':sc,'classifier':clf}
    out=HERE/'Qwen_plain_centred_heads.joblib'
    joblib.dump(bundle,out,compress=3)
    check=joblib.load(out)
    for key in ['plain','centred']:
        a,b=bundle[key],check[key]
        assert np.array_equal(a['classifier'].predict(a['scaler'].transform(z[:3])),
                              b['classifier'].predict(b['scaler'].transform(z[:3])))
    info={'artifact':out.name,'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
          'reload_parity':True,'trained_rows':len(d),'requires_local_backbone':'Qwen2.5-7B-Instruct, NF4,20 layers',
          'scope':'plain and centred full-data fitted heads; not a standalone full ensemble'}
    (HERE/'model_manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):main(Path(sys.argv[1]))
