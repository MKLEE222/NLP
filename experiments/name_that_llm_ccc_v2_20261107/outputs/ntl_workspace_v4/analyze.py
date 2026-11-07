"""Reporting-only uncertainty and feature-group audit of fixed OOF predictions."""
import json,warnings
from pathlib import Path
import numpy as np
import joblib
from method import HERE,ROOT,group_bootstrap

warnings.filterwarnings('ignore',message='X does not have valid feature names')
d=np.load(ROOT/'work'/'v4_oof_probabilities.npz',allow_pickle=True)
y,g=d['y'],d['groups']
names=[k for k in d.files if k not in ['y','groups']]
pred={k:d[k].argmax(1) for k in names}
boot=group_bootstrap(y,g,pred,np.arange(18),20261230,2000)
ug=np.unique(g)
draws=np.random.default_rng(20261231).integers(0,len(ug),size=(2000,len(ug)))
sizes=np.array([(g==v).sum() for v in ug])
loss={}
for k in names:
    p=d[k].astype(float)
    p/=p.sum(1,keepdims=True)
    point=-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1))
    totals=np.array([point[g==v].sum() for v in ug])
    loss[k]=totals[draws].sum(1)/sizes[draws].sum(1)
groups={'raw_PCA':(0,64),'residual_PCA':(64,128),'style':(128,213),
        'group_geometry':(213,217),'base_log_probability':(217,235),'base_uncertainty':(235,238)}
importance={'xgb':[],'lgb':[]}
for f in range(1,6):
    bundle=joblib.load(ROOT/'work'/f'v4_fold_{f}.joblib')
    gain=np.zeros(238)
    for key,value in bundle['boosters']['xgb'].get_booster().get_score(importance_type='total_gain').items():
        gain[int(key[1:])]=value
    importance['xgb'].append({k:float(gain[a:b].sum()/max(gain.sum(),1e-12)) for k,(a,b) in groups.items()})
    gain=bundle['boosters']['lgb'].booster_.feature_importance(importance_type='gain')
    importance['lgb'].append({k:float(gain[a:b].sum()/max(gain.sum(),1e-12)) for k,(a,b) in groups.items()})
report={'scope':'fixed-prediction reporting, no model tuning',
        'secondary_eight_way_one_sided_lower':{k:float(np.quantile(v-boot['BASE'],.05/8)) for k,v in boot.items() if k!='BASE'},
        'paired_logloss_delta_BASE_95pct':{k:np.quantile(v-loss['BASE'],[.025,.975]).tolist() for k,v in loss.items()},
        'mean_gain_fraction':{e:{k:float(np.mean([f[k] for f in rows])) for k in groups} for e,rows in importance.items()},
        'importance_scope':'gain fractions describe fitted tree splits, not causal component effects',
        'max_probability_sum_error':{k:float(np.max(np.abs(d[k].sum(1)-1))) for k in names}}
(HERE/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
