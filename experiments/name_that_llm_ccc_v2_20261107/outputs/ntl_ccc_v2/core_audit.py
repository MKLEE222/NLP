"""Use the frozen CCC interval backend on a finite OOF comparison contract."""
import sys
sys.dont_write_bytecode = True
import importlib.util
import json
import hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
source = Path(sys.argv[1] if len(sys.argv)>1 else
              'D:/科研工作区/NewThesis3/bcap_v2/experiments/aggregate_claim_compiler.py')
before = hashlib.sha256(source.read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location('frozen_ccc_aggregate', source)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
data = np.load(ROOT/'work'/'v2_oof_predictions.npz', allow_pickle=True)
y, groups, left, right = data['y'],data['groups'],data['V1'],data['B0']
different = np.flatnonzero(left!=right)
by_group = {g:np.flatnonzero(groups[different]==g) for g in np.unique(groups[different])}
actual = int(np.sign(np.sum(left==y)-np.sum(right==y)))


def execute(order):
    ls, rs = ['failure']+[None]*len(different), ['failure']+[None]*len(different)
    costs, rows = 0, 0
    for g in order:
        indices = by_group[g]
        for j in indices:
            i = different[j]
            ls[j+1] = 'success' if left[i]==y[i] else 'failure'
            rs[j+1] = 'success' if right[i]==y[i] else 'failure'
        costs += 1
        rows += int((groups==g).sum())
        a,b = core.fast_rate_interval(ls),core.fast_rate_interval(rs)
        lo,hi = a['lower']-b['upper'],a['upper']-b['lower']
        sign = 1 if lo>0 else -1 if hi<0 else 0 if lo==hi==0 else None
        if sign is not None:
            assert sign==actual, 'Invalid early publication'
            return {'prompts':costs,'rows_revealed':rows,'sign':sign,'bound':[float(lo),float(hi)]}
    raise RuntimeError('Finite comparison did not resolve')


ordered = sorted(by_group, key=lambda g:(-len(by_group[g]),str(g)))
directed = execute(ordered)
random = [execute(np.random.default_rng(s).permutation(list(by_group))) for s in range(20261200,20261300)]
assert hashlib.sha256(source.read_bytes()).hexdigest()==before
result = {'status':'COMPLETE','claim':'finite OOF ordinary-accuracy contrast V1 minus B0',
          'core_sha256':before,'core_function':'aggregate_claim_compiler.fast_rate_interval',
          'core_unchanged':True,'disagreement_rows':len(different),'actionable_prompts':len(by_group),
          'complete_accuracy_delta':float(np.mean(left==y)-np.mean(right==y)),
          'directed':directed,'random_100_mean_prompts':float(np.mean([r['prompts'] for r in random])),
          'random_100_prompt_range':[int(min(r['prompts'] for r in random)),int(max(r['prompts'] for r in random))],
          'random_100_mean_rows':float(np.mean([r['rows_revealed'] for r in random])),
          'all_publications_match_complete_audit':True,
          'scope':'fixed finite cohort; no generalization or classifier improvement certificate'}
(HERE/'core_audit_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
