"""Predict from a local fitted bundle: predict.py model.joblib input.csv output.parquet."""
import sys
from pathlib import Path
import joblib
import pandas as pd
from build_model import transform

bundle = joblib.load(sys.argv[1])
p = Path(sys.argv[2])
d = pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p)
assert {'id','text'}.issubset(d.columns)
pred = bundle['classifier'].predict(transform(bundle,d.text))
pd.DataFrame({'id':d.id,'model':pred}).to_parquet(sys.argv[3],index=False)
print(f'Wrote {len(pred)} predictions to {sys.argv[3]}')
