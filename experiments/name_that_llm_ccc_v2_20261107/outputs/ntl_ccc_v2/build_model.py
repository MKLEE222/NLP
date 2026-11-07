"""Fit a fixed candidate on all official train data and verify serialized inference."""
import sys
import json
import hashlib
import zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack,csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
from run_v2 import HERE,load_data,rich_style,style_features,lr,SEED


def transform(bundle,texts):
    blocks = [v.transform(texts) for v in bundle['vectorizers']]
    style = style_features(texts) if bundle['method']=='B0' else rich_style(texts)
    blocks.append(csr_matrix(bundle['scaler'].transform(style)*bundle['style_weight']))
    return hstack(blocks).tocsr()


def main(path,method):
    assert method in ['B0','V1','V2']
    d = load_data(path)
    vecs, blocks = [],[]
    for analyzer,ng,limit in [('char',(1,5),250000),('word',(1,2),100000)]:
        v = TfidfVectorizer(analyzer=analyzer,ngram_range=ng,min_df=2,
            max_features=limit,sublinear_tf=True,dtype=np.float32,
            token_pattern=r'(?u)\b\w+\b' if analyzer=='word' else None)
        blocks.append(v.fit_transform(d.text))
        vecs.append(v)
    style = style_features(d.text) if method=='B0' else rich_style(d.text)
    scale = StandardScaler().fit(style)
    weight = .1 if method=='B0' else .15
    x = hstack(blocks+[csr_matrix(scale.transform(style)*weight)]).tocsr()
    model = (LinearSVC(C=.5,class_weight='balanced',max_iter=5000,random_state=SEED)
             if method=='V2' else lr())
    model.fit(x,d.model)
    bundle = {'method':method,'vectorizers':vecs,'scaler':scale,'style_weight':weight,
              'classifier':model,'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    target = HERE/f'{method}_fulltrain.joblib'
    joblib.dump(bundle,target,compress=3)
    reloaded = joblib.load(target)
    with zipfile.ZipFile(path) as z:
        test = pd.read_csv(z.open('test.csv'))
        sample = pd.read_parquet(z.open('sample_submission.parquet'))
    pred = model.predict(transform(bundle,test.text))
    assert np.array_equal(pred,reloaded['classifier'].predict(transform(reloaded,test.text)))
    assert np.array_equal(test.id,sample.id)
    output = pd.DataFrame({'id':test.id,'model':pred})
    output.to_parquet(HERE/'sample_submission.parquet',index=False)
    info = {'method':method,'trained_rows':len(d),'artifact':target.name,'bytes':target.stat().st_size,
            'artifact_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'reload_predictions_equal':True,'sample_output_rows':len(test),
            'scope':'full-data fitted artifact; sample test only, no hidden-test score or Kaggle submission'}
    (HERE/'model_manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main(Path(sys.argv[1]),sys.argv[2] if len(sys.argv)>2 else 'V2')
