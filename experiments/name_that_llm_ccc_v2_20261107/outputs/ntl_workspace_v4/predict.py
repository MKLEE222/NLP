"""predict.py input.csv local_backbone model.joblib output.parquet [XGB_FULL|WORKSPACE|BASE]."""
import sys,json,warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from method import HERE,ROOT,predict
from extract_qwen import main as extract,sha

data,modeldir,headfile,out=map(Path,sys.argv[1:5])
mode=sys.argv[5] if len(sys.argv)>5 else 'XGB_FULL'
assert mode in ['WORKSPACE','BASE','NAIVE','XGB_FULL','LGB_FULL','XGB_BOUND','LGB_BOUND','NO_GUARD']
warnings.filterwarnings('ignore',message='X does not have valid feature names')
extract(data,modeldir)
d=pd.read_csv(data)
cache=ROOT/'work'/f'qwen25_7bi_nf4_l6l20_{sha(data)[:12]}.npz'
features=np.load(cache)
assert np.array_equal(features['id'],d.id.to_numpy().astype(str))
bundle=joblib.load(headfile)
meta=json.loads(cache.with_suffix('.json').read_text(encoding='utf-8'))
for k in ['checkpoint_sha256','config_sha256','layers','quantization']:
    assert meta[k]==bundle['encoder_provenance'][k]
with threadpool_limits(limits=4):p,support=predict(bundle,features['features'],d.text.to_numpy())
pd.DataFrame({'id':d.id,'model':bundle['classes'][p[mode].argmax(1)]}).to_parquet(out,index=False)
enabled=int(support.sum()) if mode=='WORKSPACE' else 0 if mode=='BASE' else len(d)
print(json.dumps({'rows':len(d),'mode':mode,'support_eligible_rows':int(support.sum()),
                  'correction_enabled_rows':enabled,'output':str(out)}))
