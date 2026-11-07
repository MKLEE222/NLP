"""Deployment identity checks added by the award-readiness audit.

Frozen research models/scores are unchanged. Existing data-only cache names
are accepted only with an encoder stamp or explicitly approved legacy hash.
"""
from pathlib import Path
import hashlib,json

TOKENIZER_FILES=['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify_encoder(modeldir,bundle,manifest):
    modeldir=Path(modeldir)
    expected=bundle['encoder_provenance']
    if file_sha(modeldir/'config.json')!=expected['config_sha256']:
        raise ValueError('Provided backbone config does not match the fitted heads; refusing cached features.')
    if file_sha(modeldir/'model.safetensors.index.json')!=expected['index_sha256']:
        raise ValueError('Provided backbone weight index does not match the fitted heads')
    source=Path(__file__).resolve().parent.parent/'ntl_public_v3'/'extract_qwen.py'
    if file_sha(source)!=manifest['encoder_source_sha256']:
        raise ValueError('Encoder implementation changed; cached feature identity must be revalidated')
    for name,value in manifest['tokenizer_sha256'].items():
        if file_sha(modeldir/name)!=value:
            raise ValueError('Provided tokenizer does not match the verified encoder: '+name)
    for name,value in expected['checkpoint_sha256'].items():
        if file_sha(modeldir/name)!=value:
            raise ValueError('Provided backbone weights do not match the fitted heads: '+name)


def validate_cache(cache,data,bundle,manifest,fresh=False):
    cache=Path(cache)
    meta=cache.with_suffix('.json')
    if not(cache.exists() and meta.exists()):
        raise ValueError('Missing or incomplete encoder feature cache')
    actual=json.loads(meta.read_text(encoding='utf-8'))
    if actual['data_sha256']!=file_sha(data):raise ValueError('Cache data identity mismatch')
    expected=bundle['encoder_provenance']
    for key in ['checkpoint_sha256','config_sha256','index_sha256','layers','truncated_layers','quantization','max_length','dimensions']:
        if actual[key]!=expected[key]:raise ValueError('Cache encoder identity mismatch: '+key)
    feature_hash=file_sha(cache)
    if actual['feature_sha256']!=feature_hash:raise ValueError('Feature cache content hash mismatch')
    fingerprint=manifest['encoder_identity_sha256']
    stamp=cache.with_suffix('.encoder-stamp.json')
    if stamp.exists():
        identity=json.loads(stamp.read_text(encoding='utf-8'))
        if identity!={'encoder_identity_sha256':fingerprint,'feature_sha256':feature_hash}:
            raise ValueError('Cache tokenizer/encoder stamp mismatch')
    elif fresh or feature_hash in manifest['approved_legacy_feature_hashes']:
        stamp.write_text(json.dumps({'encoder_identity_sha256':fingerprint,'feature_sha256':feature_hash},indent=2),encoding='utf-8')
    else:
        raise ValueError('Legacy feature cache lacks a verified tokenizer identity; regenerate it.')
