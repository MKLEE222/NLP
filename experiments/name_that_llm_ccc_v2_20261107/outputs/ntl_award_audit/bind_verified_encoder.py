"""Record current encoder identity after a fresh same-text anchor replay."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import joblib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(HERE.parent/'ntl_workspace_v4'))
from encoder_checks import TOKENIZER_FILES,file_sha,verify_encoder,validate_cache
modeldir=Path('D:/models/qwen2.5-7b')
head=HERE.parent/'ntl_workspace_v4'/'workspace_residual_model.joblib'
bundle=joblib.load(head)
old=ROOT/'work'/'qwen25_7bi_nf4_l6l20_4545aa3a3459.npz'
fresh=ROOT/'work'/'qwen25_7bi_nf4_l6l20_17bb2a9ecc42.npz'
a=np.load(old)['features'];b=np.load(fresh)['features']
rel=float(np.abs(a-b).mean()/max(np.abs(a).mean(),1e-12))
assert rel<1e-4,'Fresh encoder does not reproduce legacy anchor features'
identity={'config_sha256':bundle['encoder_provenance']['config_sha256'],
          'index_sha256':bundle['encoder_provenance']['index_sha256'],
          'checkpoint_sha256':bundle['encoder_provenance']['checkpoint_sha256'],
          'tokenizer_sha256':{n:file_sha(modeldir/n) for n in TOKENIZER_FILES},
          'encoder_source_sha256':file_sha(HERE.parent/'ntl_public_v3'/'extract_qwen.py'),
          'layers':[6,20],'truncated_layers':20,'max_length':512,
          'quantization':bundle['encoder_provenance']['quantization']}
manifest={**identity,'encoder_identity_sha256':hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),
    'head_sha256':file_sha(head),'anchor_mean_relative_feature_error':rel,
    'approved_legacy_feature_hashes':[bundle['encoder_provenance']['feature_sha256'],file_sha(old),file_sha(fresh)],
    'legacy_scope':'training artifact is the registered pinned feature hash; sample token identity checked by3 fresh anchors, not a full training re-encode'}
out=HERE.parent/'ntl_workspace_v4'/'verified_encoder_manifest.json'
verify_encoder(modeldir,bundle,manifest)
out.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
# Explicit manifest migration after a fresh anchor replay and full asset check.
assert file_sha(old)==json.loads(old.with_suffix('.json').read_text(encoding='utf-8'))['feature_sha256']
old.with_suffix('.encoder-stamp.json').write_text(json.dumps({
    'encoder_identity_sha256':manifest['encoder_identity_sha256'],'feature_sha256':file_sha(old)},indent=2),encoding='utf-8')
validate_cache(old,ROOT/'work'/'v3_official_sample.csv',bundle,manifest)
wrong_rejected=False
try:verify_encoder(Path('D:/models/qwen2.5-3b'),bundle,manifest)
except ValueError:wrong_rejected=True
assert wrong_rejected
proof={'anchor_mean_relative_feature_error':rel,'wrong_backbone_rejected_before_cache':wrong_rejected,
       'cache_content_hash_verified':True,'manifest':str(out),'scope':'deployment identity repair; no prediction retraining'}
(HERE/'encoder_fix_verification.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
print(json.dumps(proof,indent=2))
