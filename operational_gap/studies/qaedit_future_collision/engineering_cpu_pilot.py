"""CPU-only engineering feasibility pilot. NOT the frozen scientific experiment.

Uses a *single* persistent LoRA state in an actual open Qwen 0.5B model.
Only the first precommitted development pair/next action is exposed.
No confirmation pool is evaluated. Output hashes and aggregated metrics only.
"""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from prepare_manifest import load_rows, clean_record, build

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_SHA = "a338b55dd21219a5f4da42bc11a9313d1a27d4cc"
SEED = 20261011

def h(v):
    return hashlib.sha256(v.encode("utf-8")).hexdigest()[:16]

def canonical(text):
    return " ".join(text.strip().casefold().split())

def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    print("ENV cpu_threads=2 torch=", torch.__version__, "torch_cuda=", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        raise RuntimeError("This is explicitly a CPU feasibility pilot")
    data, filehash = load_rows(args.dataset)
    manifest = build(data, filehash)
    selected = manifest["dev"]
    pair = selected["trajectories"][0]
    future = selected["events"][0]
    assert pair["pair"] == future["pair"] == 0
    # One edit on each side: off-protocol cost-constrained feasibility run,
    # not the registered 8-/16-edit check point.
    ids = [pair["A"][0], pair["B"][0], future["future_action_id"]]
    lookup = {}
    for row in data:
        c = clean_record(row)
        if c and c["uid"] in ids:
            lookup[c["uid"]] = c
    assert len(lookup) == 3
    print("DATA file_sha256=", filehash, "three_id_hashes=", [h(uid) for uid in ids], flush=True)
    del data

    t0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_SHA, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_SHA, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=False)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj","v_proj"],
        lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM))
    print("MODEL revision=", MODEL_SHA,
          "trainable_params=", sum(p.numel() for p in model.parameters() if p.requires_grad),
          "loaded_sec=", round(time.monotonic()-t0,2), flush=True)

    trainable = {name:p for name,p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in name for name in trainable), "Unexpected trainable base weights"

    def snapshot():
        return {name:p.detach().clone() for name,p in trainable.items()}

    def restore(state):
        with torch.no_grad():
            for name, value in state.items():
                trainable[name].copy_(value)

    def training_sequence(question, answer):
        prefix = tokenizer.apply_chat_template(
            [{"role":"user","content":question}],
            add_generation_prompt=True, tokenize=True)
        full = tokenizer.apply_chat_template(
            [{"role":"user","content":question},{"role":"assistant","content":answer}],
            add_generation_prompt=False, tokenize=True)
        if not isinstance(full,list) or not isinstance(prefix,list):
            raise ValueError("Unexpected chat-template encoding")
        if full[:len(prefix)] != prefix:
            raise ValueError("Assistant prefix not stable under template")
        if len(full) > 320 or len(prefix) >= len(full):
            raise ValueError("Training sequence violates frozen max length or has no target")
        input_ids = torch.tensor([full], dtype=torch.long)
        labels = input_ids.clone()
        labels[:, :len(prefix)] = -100
        return input_ids, labels

    def update(row):
        inputs, labels = training_sequence(row["prompt"],row["target"])
        model.train()
        opt = torch.optim.AdamW(trainable.values(),lr=2e-4,weight_decay=0.0)
        losses=[]
        for i in range(8):
            opt.zero_grad(set_to_none=True)
            out = model(input_ids=inputs, labels=labels)
            if not torch.isfinite(out.loss):
                raise RuntimeError("Nonfinite training loss")
            out.loss.backward()
            opt.step()
            losses.append(round(float(out.loss.item()),5))
        del opt, inputs, labels
        return losses

    def gen(question):
        model.eval()
        prefix=tokenizer.apply_chat_template(
            [{"role":"user","content":question}],
            add_generation_prompt=True, tokenize=True)
        inp=torch.tensor([prefix],dtype=torch.long)
        with torch.inference_mode():
            output=model.generate(
                input_ids=inp, do_sample=False, max_new_tokens=48,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True)
        return tokenizer.decode(output[0, len(prefix):], skip_special_tokens=True)

    def evaluate(row):
        values=[]
        items=[("target",row["prompt"],row["target"]),
               ("rephrase",row["rephrase"],row["target"]),
               ("locality",row["locality"][0]["loc"],row["locality"][0]["loc_ans"])]
        for name, q, gold in items:
            ans=gen(q)
            values.append({"metric":name,"correct":int(canonical(ans)==canonical(gold)),
                           "response_fingerprint":h(canonical(ans))})
        return values

    start_state=snapshot()
    # Two actual sequential history states with common adapter initialization.
    snapshots=[]
    history_losses=[]
    for side, uid in zip(("A","B"),ids[:2]):
        restore(start_state)
        losses=update(lookup[uid])
        snapshots.append(snapshot())
        history_losses.append({"side":side, "losses":losses})
        print("HISTORY",side,"loss_first_last",losses[0],losses[-1],
              "elapsed_sec",round(time.monotonic()-t0,2),flush=True)

    results=[]
    future_row=lookup[ids[2]]
    for side, trained in zip(("A","B"),snapshots):
        restore(trained)
        # Preflight contract Q0 = the future edit question only (read only).
        q0=gen(future_row["prompt"])
        before=evaluate(future_row)
        after_loss=update(future_row)
        after=evaluate(future_row)
        deltas=[int(y["correct"])-int(x["correct"]) for x,y in zip(before,after)]
        results.append({"side":side,
                        "preflight_Q0_hash":h(canonical(q0)),
                        "before":before, "after":after,
                        "delta_H":deltas,
                        "future_loss_first_last":[after_loss[0],after_loss[-1]]})
        print("NEXT",side,"q0hash",h(canonical(q0)),"deltas",deltas,
              "elapsed_sec",round(time.monotonic()-t0,2),flush=True)

    collision=results[0]["preflight_Q0_hash"]==results[1]["preflight_Q0_hash"]
    divergence=max(abs(a-b) for a,b in zip(results[0]["delta_H"],results[1]["delta_H"]))
    report={
        "status":"ENGINEERING_FEASIBILITY_ONLY_NOT_REGISTERED_CONFIRMATION",
        "source_sha256":filehash,
        "source_commit":"61ccca3c1ecdb3976c9c0924e9352f3e64cc7d48",
        "model":MODEL_ID,"model_revision":MODEL_SHA,
        "seed":SEED,"train_steps_per_edit":8,"lr":2e-4,
        "lora_rank":8,"lora_alpha":16,
        "history_edits_each":1,"future_actions":1,
        "checkpoint_off_protocol":"one edit per side",
        "preflight_budget":"Q0 prospective read only",
        "history_loss":history_losses,
        "results":results,
        "preflight_Q0_match":collision,
        "observed_future_consequence_divergence":divergence,
        "elapsed_sec":round(time.monotonic()-t0,2),
        "meaning":"One engineering pilot event only. No statistical or novelty inference.",
    }
    Path(args.report).write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("PILOT_RESULT",json.dumps({
        "q0_match":collision,"future_delta_divergence":divergence,
        "history_steps":8, "future_steps":8,
        "elapsed_sec":report["elapsed_sec"],"conclusion":"INCONCLUSIVE_PILOT"
    }),flush=True)


if __name__ == "__main__":
    main()
