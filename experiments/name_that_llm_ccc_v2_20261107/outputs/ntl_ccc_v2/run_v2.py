"""Fixed nested prompt-group evaluation; see PROTOCOL.md."""
from pathlib import Path
import sys
import json
import hashlib
import time
import re
import platform
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / 'ntl_ccc_killtest'))
from run_killtest import load_data, style_features, score, recall_vector

SEED = 20261110
REV = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
MODEL_PATH = Path('C:/Users/25317/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots') / REV


def rich_style(texts):
    base = style_features(texts)
    extra = []
    for raw in texts:
        s = str(raw)
        words = re.findall(r"\b[\w']+\b", s.lower())
        lines = s.splitlines()
        paras = re.split(r'\n\s*\n', s.strip())
        sentences = [x for x in re.split(r'[.!?]+\s+', s) if x.strip()]
        n, nw = max(len(s), 1), max(len(words), 1)
        lens = np.array([len(x.split()) for x in sentences] or [0])
        freqs = np.array(list(pd.Series(words, dtype=str).value_counts()), dtype=float)
        probs = freqs / max(freqs.sum(), 1)
        row = [len(set(words))/nw, len(paras), np.mean(lens), np.std(lens),
               np.max(lens), np.median(lens), np.mean([len(x) for x in lines] or [0]),
               -float(np.sum(probs*np.log(np.maximum(probs, 1e-12)))),
               sum(len(w)>=10 for w in words)/nw,
               sum(len(w)<=3 for w in words)/nw,
               sum(w.istitle() for w in s.split())/nw]
        row += [s.count(ch)/n for ch in ['(',')','[',']','{','}','*','#','`','/','\\','%','&','+','=','—','–','…','“','”',"'",'\n','\t']]
        row += [len(re.findall(p, s, flags=re.M))/max(len(lines),1) for p in
                [r'^\s*[-*+]\s', r'^\s*\d+[.)]\s', r'^\s*#{1,6}\s', r'^\s*>', r'^\s*\|', r'^\s*$']]
        row += [words.count(w)/nw for w in ['the','and','to','of','a','in','is','it','you','i','we','that','this','with','for','can','however','also','but','so','therefore','certainly','here','overall']]
        row += [len(re.findall(p, s, flags=re.I))/nw for p in
                [r'\bas an?\b',r'\blet.s\b',r'\bfor example\b',r'\bin summary\b',r'\bit.s\b',r'\byou can\b']]
        extra.append(row)
    return np.hstack([base, np.asarray(extra)])


def embeddings(texts, data_hash):
    cache = ROOT/'work'/f'minilm_{data_hash[:12]}_{REV[:8]}.npy'
    if cache.exists():
        return np.load(cache), {'revision': REV, 'cache': True}
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = SentenceTransformer(str(MODEL_PATH), device=device, local_files_only=True)
    m.max_seq_length = 256
    views = []
    for text in texts:
        words = text.split()
        views += [' '.join(words[:160]), ' '.join(words[-160:])]
    try:
        z = m.encode(views, batch_size=64 if device=='cuda' else 32, show_progress_bar=True,
                     normalize_embeddings=True, convert_to_numpy=True)
    except RuntimeError as e:
        if device != 'cuda':
            raise
        print('GPU encoding unavailable; retry on CPU:', str(e)[:200], flush=True)
        device = 'cpu'
        m.to('cpu')
        z = m.encode(views, batch_size=32, show_progress_bar=True,
                     normalize_embeddings=True, convert_to_numpy=True)
    z = z.reshape(len(texts), 2, -1).mean(axis=1)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    np.save(cache, z)
    return z, {'revision': REV, 'device': device, 'cache': False,
               'config_sha256': hashlib.sha256((MODEL_PATH/'config.json').read_bytes()).hexdigest()}


def lr(c=2):
    return LogisticRegression(C=c, max_iter=600, class_weight='balanced', random_state=SEED)


def fit_predict(texts, oldstyle, newstyle, emb, y, tr, va):
    train_blocks, eval_blocks = [], []
    for analyzer, ng, limit in [('char',(1,5),250000), ('word',(1,2),100000)]:
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ng, min_df=2,
                              max_features=limit, sublinear_tf=True, dtype=np.float32,
                              token_pattern=r'(?u)\b\w+\b' if analyzer=='word' else None)
        train_blocks.append(vec.fit_transform(texts[tr]))
        eval_blocks.append(vec.transform(texts[va]))
    xtr, xva = {}, {}
    for name, features, weight in [('B0',oldstyle,.1), ('V1',newstyle,.15)]:
        scale = StandardScaler().fit(features[tr])
        xtr[name] = hstack(train_blocks+[csr_matrix(scale.transform(features[tr])*weight)]).tocsr()
        xva[name] = hstack(eval_blocks+[csr_matrix(scale.transform(features[va])*weight)]).tocsr()
    b = lr().fit(xtr['B0'], y[tr])
    rich = lr().fit(xtr['V1'], y[tr])
    prob = rich.predict_proba(xva['V1'])
    pred = {'B0': b.predict(xva['B0']), 'V1': rich.classes_[prob.argmax(axis=1)]}
    svm = LinearSVC(C=.5, class_weight='balanced', max_iter=5000, random_state=SEED)
    pred['V2'] = svm.fit(xtr['V1'], y[tr]).predict(xva['V1'])
    if emb is not None:
        model = lr(1).fit(emb[tr], y[tr])
        assert list(model.classes_) == list(rich.classes_)
        p = .85*prob + .15*model.predict_proba(emb[va])
        pred['V3'] = rich.classes_[p.argmax(axis=1)]
    return pred


def group_bootstrap(y, groups, preds, classes, seed, count):
    ug = np.unique(groups)
    totals = np.zeros((len(ug),len(classes)))
    correct = {k: totals.copy() for k in preds}
    for i, g in enumerate(ug):
        for j, c in enumerate(classes):
            ix = (groups==g)&(y==c)
            totals[i,j] = ix.sum()
            for k, pred in preds.items():
                correct[k][i,j] = (pred[ix]==y[ix]).sum()
    draws = np.random.default_rng(seed).integers(0,len(ug),size=(count,len(ug)))
    denom = totals[draws].sum(axis=1)
    return {k: np.nanmean(np.divide(v[draws].sum(axis=1),denom,
                    out=np.full_like(denom,np.nan),where=denom>0),axis=1) for k,v in correct.items()}


def main(path):
    start = time.time()
    d = load_data(path)
    dh = hashlib.sha256(path.read_bytes()).hexdigest()
    texts, y, groups = d.text.to_numpy(), d.model.to_numpy(), d.prompt_id.to_numpy()
    classes = np.asarray(sorted(d.model.unique()))
    styles, richer = style_features(texts), rich_style(texts)
    print('Rows',len(d),'rich style dimensions',richer.shape[1],flush=True)
    try:
        emb, meta = embeddings(texts, dh)
    except Exception as e:
        emb, meta = None, {'status':'BLOCKED','error':repr(e)}
        print('V3 blocked',meta,flush=True)
    names = ['B0','V1','V2'] + (['V3'] if emb is not None else [])
    oof = {k: np.full(len(d),'',dtype=object) for k in names+['SELECTED']}
    fold_ids = np.full(len(d),-1)
    folds = np.array_split(np.random.default_rng(SEED).permutation(sorted(np.unique(groups))),5)
    logs = []
    for fold, val_groups in enumerate(folds):
        tr_groups = np.asarray(sorted(set(groups)-set(val_groups)))
        shuffled = np.random.default_rng(SEED+fold).permutation(tr_groups)
        inner_va = np.flatnonzero(np.isin(groups,shuffled[:24]))
        inner_tr = np.flatnonzero(np.isin(groups,shuffled[24:]))
        outer_tr = np.flatnonzero(np.isin(groups,tr_groups))
        outer_va = np.flatnonzero(np.isin(groups,val_groups))
        assert not (set(groups[outer_tr]) & set(groups[outer_va]))
        print('Fold',fold+1,'inner fit',flush=True)
        ip = fit_predict(texts,styles,richer,emb,y,inner_tr,inner_va)
        scores = {k:score(y[inner_va],p,classes) for k,p in ip.items()}
        boot = group_bootstrap(y[inner_va],groups[inner_va],ip,classes,20261120+fold,1000)
        lower = {k:float(np.quantile(boot[k]-boot['B0'],.05/3)) for k in names if k!='B0'}
        passed = [k for k in lower if lower[k]>0]
        chosen = max(passed,key=lambda k:scores[k]) if passed else 'B0'
        print('Fold',fold+1,'choice',chosen,'outer fit',flush=True)
        op = fit_predict(texts,styles,richer,emb,y,outer_tr,outer_va)
        for k,p in op.items():
            oof[k][outer_va] = p
        oof['SELECTED'][outer_va] = op[chosen]
        fold_ids[outer_va] = fold
        log = {'fold':fold+1,'inner_BA':scores,'inner_delta_lower':lower,'choice':chosen,
               'outer_BA':{k:score(y[outer_va],p,classes) for k,p in op.items()},
               'outer_rows':len(outer_va)}
        logs.append(log)
        print(json.dumps(log),flush=True)
    assert (fold_ids>=0).all() and all((p!='').all() for p in oof.values())
    boot = group_bootstrap(y,groups,oof,classes,20261130,2000)
    result = {'status':'COMPLETE','data_sha256':dh,'seed':SEED,'folds':logs,
              'OOF_balanced_accuracy':{k:score(y,p,classes) for k,p in oof.items()},
              'OOF_intervals':{k:{'BA_95pct':np.quantile(v,[.025,.975]).tolist(),
                   'paired_delta_B0_95pct':np.quantile(v-boot['B0'],[.025,.975]).tolist()}
                   for k,v in boot.items()},
              'class_recall':{k:dict(zip(classes,recall_vector(y,p,classes))) for k,p in oof.items()},
              'embedding':meta,'rich_style_dimensions':richer.shape[1],
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol_sha256':hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),
              'seconds':round(time.time()-start,2),'python':platform.python_version(),
              'scope':'internal nested prompt CV after iteration1; not untouched external test'}
    (HERE/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(ROOT/'work'/'v2_oof_predictions.npz',y=y,groups=groups,fold=fold_ids,**oof)
    print(json.dumps({k:result[k] for k in ['OOF_balanced_accuracy','OOF_intervals','seconds']},indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main(Path(sys.argv[1] if len(sys.argv)>1 else 'D:/name-that-llm.zip'))
