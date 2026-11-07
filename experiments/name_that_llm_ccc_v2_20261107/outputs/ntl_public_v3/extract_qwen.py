"""Frozen decoder feature extraction only; no generation or label fitting."""
import sys
import json
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig,AutoModel,AutoTokenizer,BitsAndBytesConfig,logging

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(HERE.parent/'ntl_ccc_killtest'))
from run_killtest import load_data


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main(data,modeldir):
    start=time.time()
    torch.set_num_threads(4)
    d=load_data(data)
    dh=sha(data)
    cache=ROOT/'work'/f'qwen25_7bi_nf4_l6l20_{dh[:12]}.npz'
    meta=cache.with_suffix('.json')
    if cache.exists() and meta.exists():
        print('Existing exact-data feature cache',cache,flush=True)
        return
    assert torch.cuda.is_available()
    cfg=AutoConfig.from_pretrained(modeldir,local_files_only=True)
    cfg.num_hidden_layers=20
    cfg.use_cache=False
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
          bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    logging.set_verbosity_error()
    print('Loading local Qwen 7B-Instruct,20 layers,NF4',flush=True)
    model=AutoModel.from_pretrained(modeldir,config=cfg,quantization_config=quant,
          dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',
          local_files_only=True).eval()
    tok=AutoTokenizer.from_pretrained(modeldir,local_files_only=True)
    tok.padding_side='right'
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    texts=d.text.to_numpy()
    order=np.argsort([len(s) for s in texts],kind='stable')
    features=np.zeros((len(texts),cfg.hidden_size*2),dtype=np.float32)
    batch_size=4
    for b0 in range(0,len(texts),batch_size):
        ix=order[b0:b0+batch_size]
        enc=tok([(tok.bos_token or '')+texts[i] for i in ix],return_tensors='pt',padding=True,
                truncation=True,max_length=512,add_special_tokens=False).to('cuda')
        with torch.inference_mode():
            hs=model(**enc,output_hidden_states=True,use_cache=False).hidden_states
            mask=enc['attention_mask'].float().unsqueeze(-1)
            z=torch.cat([(hs[layer].float()*mask).sum(1)/mask.sum(1).clamp(min=1)
                         for layer in (6,20)],dim=1)
        if not torch.isfinite(z).all(): raise ValueError('Nonfinite backbone feature')
        features[ix]=z.cpu().numpy()
        del hs,z,enc
        if b0%(batch_size*25)==0:
            print('Encoded',min(b0+batch_size,len(texts)),'/',len(texts),
                  'elapsed',round(time.time()-start,1),flush=True)
    np.savez_compressed(cache,features=features,id=d.id.to_numpy().astype(str))
    print('Hashing local checkpoint for provenance',flush=True)
    info={'data_sha256':dh,'model':'Qwen2.5-7B-Instruct','layers':[6,20],
          'truncated_layers':20,'quantization':'NF4 double quantization bf16 compute',
          'max_length':512,'rows':len(d),'dimensions':features.shape[1],
          'checkpoint_sha256':{p.name:sha(p) for p in sorted(modeldir.glob('*.safetensors'))},
          'config_sha256':sha(modeldir/'config.json'),'index_sha256':sha(modeldir/'model.safetensors.index.json'),
          'feature_sha256':sha(cache),'seconds':round(time.time()-start,2)}
    meta.write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':
    main(Path(sys.argv[1]),Path(sys.argv[2]))
