"""Workspace grouped residual correction. See the frozen protocol."""
from pathlib import Path
import sys
import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import xgboost as xgb
import lightgbm as lgb

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(HERE.parent/'ntl_public_v3'))
sys.path.insert(0,str(HERE.parent/'ntl_ccc_v2'))
from public_stylegram import normalised_content_groups,eligible
from run_v2 import rich_style,load_data,score,group_bootstrap,recall_vector
from run_killtest import by_prompt_center


def logp(p): return np.log(np.clip(p,1e-8,1.))


def context(z,texts):
    inferred=normalised_content_groups(texts)
    valid=eligible(inferred)
    centered=np.zeros_like(z)
    centered[valid]=by_prompt_center(z[valid],inferred[valid])
    diag=np.zeros((len(z),4),dtype=np.float32)
    for g in np.unique(inferred):
        ix=np.flatnonzero(inferred==g)
        centroid=z[ix].mean(0)
        norms=np.linalg.norm(z[ix],axis=1)
        cosine=z[ix]@centroid/np.maximum(norms*np.linalg.norm(centroid),1e-8)
        diag[ix,0]=np.log1p(len(ix))
        diag[ix,1]=cosine
        diag[ix,2]=np.log1p(norms)
        diag[ix,3]=np.linalg.norm(centered[ix],axis=1)/np.maximum(norms,1e-8)
    return {'centered':centered,'diag':diag,'valid':valid}


def fit_base(z,y,g,ix):
    result={}
    for name,x in [('raw',z[ix]),('centered',by_prompt_center(z[ix],g[ix]))]:
        sc=StandardScaler().fit(x)
        m=LogisticRegression(C=.0025,max_iter=2000,class_weight='balanced',random_state=42)
        m.fit(sc.transform(x),y[ix])
        assert np.array_equal(m.classes_,np.arange(18))
        result[name]=(sc,m)
    return result


def predict_base(base,z,t,ctx=None):
    ctx=context(z,t) if ctx is None else ctx
    sc,m=base['raw']
    p=m.predict_proba(sc.transform(z))
    use=ctx['valid']
    sc,m=base['centered']
    if use.any():p[use]=m.predict_proba(sc.transform(ctx['centered'][use]))
    return p,ctx


def crossfit_base(z,y,g,t,ix,seed):
    groups=np.random.default_rng(seed).permutation(sorted(np.unique(g[ix])))
    p=np.zeros((len(ix),18))
    ctx={'centered':np.zeros_like(z[ix]),'diag':np.zeros((len(ix),4),np.float32),
         'valid':np.zeros(len(ix),bool)}
    for fold,hold in enumerate(np.array_split(groups,3)):
        keep=~np.isin(g[ix],hold)
        fit,va=ix[keep],ix[~keep]
        assert not(set(g[fit])&set(g[va]))
        b=fit_base(z,y,g,fit)
        pp,cc=predict_base(b,z[va],t[va])
        p[~keep]=pp
        for k in ctx:ctx[k][~keep]=cc[k]
    assert (p.sum(1)>.99).all()
    return p,ctx


class Projection:
    def fit(self,z):
        self.sc=StandardScaler().fit(z)
        self.pca=PCA(n_components=64,svd_solver='randomized',random_state=20261210).fit(self.sc.transform(z))
        raw=self.pca.transform(self.sc.transform(z))
        self.limit=float(np.quantile(np.linalg.norm(raw,axis=1),.99))
        return self

    def transform(self,z,styles,ctx,p):
        raw=self.pca.transform(self.sc.transform(z))
        centered=(ctx['centered']/self.sc.scale_)@self.pca.components_.T
        rep=np.hstack([raw,centered,styles,ctx['diag']]).astype(np.float32)
        lp=logp(p)
        sort=np.sort(p,axis=1)
        uncertainty=np.column_stack([-(p*lp).sum(1),sort[:,-1],sort[:,-1]-sort[:,-2]])
        stacked=np.hstack([rep,lp,uncertainty]).astype(np.float32)
        support=ctx['valid']&(np.linalg.norm(raw,axis=1)<=self.limit)
        return rep,stacked,support


def new_xgb(rounds,early=False):
    args=dict(n_estimators=rounds,max_depth=3,min_child_weight=10,gamma=.1,
        learning_rate=.05,subsample=.8,colsample_bytree=.8,reg_alpha=1,reg_lambda=20,
        max_bin=63,n_jobs=4,tree_method='hist',objective='multi:softprob',num_class=18,
        eval_metric='mlogloss',random_state=42,device='cpu')
    if early:args['early_stopping_rounds']=30
    return xgb.XGBClassifier(**args)


def new_lgb(rounds):
    return lgb.LGBMClassifier(n_estimators=rounds,num_leaves=7,max_depth=3,
        min_child_samples=60,min_child_weight=1,learning_rate=.05,
        subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_alpha=1,reg_lambda=20,
        max_bin=63,n_jobs=4,objective='multiclass',num_class=18,verbosity=-1,
        deterministic=True,force_col_wise=True,random_state=42)


def fit_boosters(rep,stacked,y,p,cal=None,rounds=None):
    early=cal is not None
    counts={'plain':400,'xgb':400,'lgb':400} if early else rounds
    plain=new_xgb(counts['plain'],early)
    xb=new_xgb(counts['xgb'],early)
    lb=new_lgb(counts['lgb'])
    if early:
        cr,cs,cy,cp=cal
        plain.fit(rep,y,eval_set=[(cr,cy)],verbose=False)
        xb.fit(stacked,y,base_margin=logp(p),eval_set=[(cs,cy)],
               base_margin_eval_set=[logp(cp)],verbose=False)
        lb.fit(stacked,y,init_score=logp(p),eval_set=[(cs,cy)],eval_init_score=[logp(cp)],
               eval_metric='multi_logloss',callbacks=[lgb.early_stopping(30,verbose=False)])
        counts={'plain':int(plain.best_iteration)+1,'xgb':int(xb.best_iteration)+1,
                'lgb':int(lb.best_iteration_)}
        xfull=softmax(xb.predict(cs,base_margin=logp(cp),output_margin=True,
                             iteration_range=(0,counts['xgb'])),axis=1)
        lfull=softmax(logp(cp)+lb.predict(cs,raw_score=True,num_iteration=counts['lgb']),axis=1)
        assert abs(log_loss(cy,xfull,labels=np.arange(18))-float(xb.best_score))<1e-5
        assert abs(log_loss(cy,lfull,labels=np.arange(18))-lb.best_score_['valid_0']['multi_logloss'])<1e-5
    else:
        plain.fit(rep,y,verbose=False)
        xb.fit(stacked,y,base_margin=logp(p),verbose=False)
        lb.fit(stacked,y,init_score=logp(p))
    return {'plain':plain,'xgb':xb,'lgb':lb,'rounds':counts}


def combine(models,rep,stacked,p,support):
    rounds=models['rounds']
    plain=models['plain'].predict_proba(rep,iteration_range=(0,rounds['plain']))
    xd=models['xgb'].predict(stacked,base_margin=logp(p),output_margin=True,
                            iteration_range=(0,rounds['xgb']))-logp(p)
    ld=models['lgb'].predict(stacked,raw_score=True,num_iteration=rounds['lgb'])
    delta=np.clip((xd+ld)/2,-1,1)
    return {'BASE':p,'TREE':plain,'NAIVE':.75*p+.25*plain,
        'XGB_FULL':softmax(logp(p)+xd,axis=1),'LGB_FULL':softmax(logp(p)+ld,axis=1),
        'XGB_BOUND':softmax(logp(p)+.25*np.clip(xd,-1,1),axis=1),
        'LGB_BOUND':softmax(logp(p)+.25*np.clip(ld,-1,1),axis=1),
        'NO_GUARD':softmax(logp(p)+.25*delta,axis=1),
        'WORKSPACE':softmax(logp(p)+.25*delta*support[:,None],axis=1)}


def predict(bundle,z,texts):
    p,ctx=predict_base(bundle['base'],z,texts)
    rep,stacked,support=bundle['projection'].transform(z,rich_style(texts),ctx,p)
    return combine(bundle['boosters'],rep,stacked,p,support),support
