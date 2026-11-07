"""Predict id/model parquet using a local backbone and exported fixed heads."""
import sys,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from extract_qwen import main as extract,sha,ROOT
from public_stylegram import normalised_content_groups,eligible
from run_killtest import by_prompt_center

data,modeldir,headfile,out=map(Path,sys.argv[1:5])
extract(data,modeldir)
d=pd.read_csv(data)
dh=sha(data)
cache=ROOT/'work'/f'qwen25_7bi_nf4_l6l20_{dh[:12]}.npz'
f=np.load(cache)
assert np.array_equal(f['id'],d.id.to_numpy().astype(str))
z=f['features']
b=joblib.load(headfile)
for key in ['config_sha256','checkpoint_sha256']:
    assert json.loads(cache.with_suffix('.json').read_text(encoding='utf-8'))[key]==b['feature_provenance'][key]
p=b['plain']['classifier'].predict_proba(b['plain']['scaler'].transform(z))
try:
    groups=normalised_content_groups(d.text.to_numpy())
except ValueError:
    groups=np.arange(1,len(d)+1)
use=eligible(groups)
if use.any():
    p[use]=b['centred']['classifier'].predict_proba(b['centred']['scaler'].transform(
            by_prompt_center(z[use],groups[use])))
pd.DataFrame({'id':d.id,'model':b['classes'][p.argmax(1)]}).to_parquet(out,index=False)
print(json.dumps({'rows':len(d),'group_centred_rows':int(use.sum()),'output':str(out)}))
