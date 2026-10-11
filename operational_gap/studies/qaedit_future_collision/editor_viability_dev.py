"""Development-only single-edit viability screen; no confirmation records opened.

Tests *only* whether the frozen LoRA recipe can actually improve free-generation
accuracy on the first eight preregistered development-history QAEdit records.
Resetting the adapter between rows is intentional and means this is NOT a
sequential-edit result or a test of consequence collision.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from prepare_manifest import clean_record, load_rows, build


def normalize(s):
    return " ".join(s.casefold().strip().split())


def hashshort(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--report",required=True)
    a=p.parse_args()

    seed=20261011
    random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    rows, sha=load_rows(a.dataset)
    manifest=build(rows,sha)
    subset=manifest["dev"]["trajectories"][0]["A"][:8]
    samples={}
    for r in rows:
        item=clean_record(r)
        if item and item["uid"] in subset:
            samples[item["uid"]]=item
    if len(samples)!=len(subset):
        raise RuntimeError("Missing frozen sample records")
    del rows

    model_id="Qwen/Qwen2.5-0.5B-Instruct"
    model_sha="a338b55dd21219a5f4da42bc11a9313d1a27d4cc"
    t0=time.monotonic()
    tok=AutoTokenizer.from_pretrained(model_id,revision=model_sha)
    model=AutoModelForCausalLM.from_pretrained(
        model_id,revision=model_sha,torch_dtype=torch.float32,low_cpu_mem_usage=True,
        trust_remote_code=False)
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(
        r=8,lora_alpha=16,target_modules=["q_proj","v_proj"],
        lora_dropout=0,bias="none",task_type=TaskType.CAUSAL_LM))
    params={name:p for name,p in model.named_parameters() if p.requires_grad}
    assert params and all("lora_" in name for name in params)
    initial={name:p.detach().clone() for name,p in params.items()}
    print("MODEL",model_id,model_sha,"trainable",sum(p.numel() for p in params.values()),flush=True)

    def restore():
        with torch.no_grad():
            for name,value in initial.items():
                params[name].copy_(value)

    def gen(q):
        model.eval()
        ids=tok.apply_chat_template([{"role":"user","content":q}],
                                    add_generation_prompt=True,tokenize=True)
        inp=torch.tensor([ids])
        with torch.inference_mode():
            result=model.generate(input_ids=inp,do_sample=False,max_new_tokens=48,
                                  use_cache=True,pad_token_id=tok.eos_token_id)
        return tok.decode(result[0,len(ids):],skip_special_tokens=True)

    def train(q,target):
        pre=tok.apply_chat_template([{"role":"user","content":q}],
                                    add_generation_prompt=True,tokenize=True)
        seq=tok.apply_chat_template(
            [{"role":"user","content":q},{"role":"assistant","content":target}],
            add_generation_prompt=False,tokenize=True)
        if seq[:len(pre)]!=pre or len(seq)>320:
            raise ValueError("Frozen edit violates 320 token bound/chat template")
        x=torch.tensor([seq]); y=x.clone();y[:,:len(pre)]=-100
        opt=torch.optim.AdamW(params.values(),lr=2e-4,weight_decay=0)
        model.train()
        losses=[]
        for _ in range(8):
            opt.zero_grad(set_to_none=True)
            loss=model(input_ids=x,labels=y).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite edit loss")
            loss.backward();opt.step()
            losses.append(round(float(loss.item()),5))
        del opt
        return losses

    results=[]
    for i,uid in enumerate(subset):
        item=samples[uid]
        restore()
        before=gen(item["prompt"])
        losses=train(item["prompt"],item["target"])
        after=gen(item["prompt"])
        gold=normalize(item["target"])
        b=normalize(before); aft=normalize(after)
        res={
            "index":i,"id_sha16":hashshort(uid),"source":item["source"],
            "was_correct":int(b==gold),
            "now_correct":int(aft==gold),
            "strict_gain":int(aft==gold)-int(b==gold),
            "loose_gold_in_after":int(bool(gold) and gold in aft),
            "generation_changed":b!=aft,
            "training_loss_first_last":[losses[0],losses[-1]]
        }
        results.append(res)
        print("VIABILITY_ROW",json.dumps(res,sort_keys=True),
              "elapsed",round(time.monotonic()-t0,2),flush=True)

    initially_wrong=[x for x in results if x["was_correct"]==0]
    improved=sum(x["now_correct"] for x in initially_wrong)
    viability_gate=("PASS_SMOKE" if len(initially_wrong)>=4 and improved>=2
                    else "BLOCK_CURRENT_EDITOR_FOR_FULL_CONFIRMATION")
    report={
        "status":"DEVELOPMENT_ONLY_OFF_REGISTERED_SEQUENTIAL_PROTOCOL",
        "source_sha256":sha,"model":model_id,"revision":model_sha,
        "sample_selection":"first 8 ID-frozen history-A development records",
        "configuration":{"steps":8,"lora_r":8,"alpha":16,
                         "lr":0.0002,"max_train_tokens":320,"max_new_tokens":48},
        "results":results,
        "summary":{"n":len(results),"initially_wrong":len(initially_wrong),
                   "strict_post_improvements":improved,
                   "generation_changed":sum(x["generation_changed"] for x in results),
                   "loose_answer_in_after":sum(x["loose_gold_in_after"] for x in results)},
        "engineering_gate":viability_gate,
        "timing_seconds":round(time.monotonic()-t0,2),
        "evidence_limit":"Single-edit viability only; no sequential collision or generalization inference."
    }
    Path(a.report).write_text(json.dumps(report,indent=2)+"\n")
    print("VIABILITY_RESULT",json.dumps({
        "engineering_gate":viability_gate,"summary":report["summary"],
        "elapsed_sec":report["timing_seconds"]}),flush=True)


if __name__=="__main__":
    main()
