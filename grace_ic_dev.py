import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 20260923
GRACE_COMMIT = "f674183f17a995d109e10ee6140d4c3e6d016115"
MODEL_ID = "tomh/scotus-bert"
TOKENIZER_ID = "bert-base-cased"
DATASET_ID = "tomh/grace-scotus"
SALT = "GRACE-IC-20260923"
TARGET_KEYS = 4
MAX_ANCHOR_EDITS = 12
ANCHOR_CANDIDATES = 128
PROBE_CANDIDATES = 256
DEV_CANDIDATES = 256
HELDOUT_CANDIDATES = 256
DEV_CONSEQUENCE_N = 16
PROBE_COUNT = 4
ORDERS = (1, 2, 3)
U_GRID = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)
ETA_GRID = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
RANK_REL = 1e-4
RANK_DIAG = (1e-3, 1e-5)
PROJECTOR_TOL = 0.05
SIGNAL_RMS_MIN = 1e-4
CURVE_NRMSE_MAX = 0.05
MAX_ABS_RESPONSE = 0.5
GRACE_LAYER = "bert.encoder.layer[10].output.dense.weight"


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def stable_key(example):
    return sha_text(f"{SALT}|{int(example['label'])}|{example['text']}")


def tensor_sha(t: torch.Tensor) -> str:
    a = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def normalized_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    gold = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    total = logits.sum(dim=1)
    other_mean = (total - gold) / (logits.shape[1] - 1)
    return gold - other_mean


def hadamard4(device, dtype):
    h = torch.tensor(
        [[1, 1, 1, 1],
         [1, -1, 1, -1],
         [1, 1, -1, -1],
         [1, -1, -1, 1]],
        dtype=dtype,
        device=device,
    ) / 2.0
    return h


def load_grace_class(grace_repo: Path):
    # grace.utils imports wandb, but this executor does not use logging;
    # a module stub avoids pulling an unrelated runtime dependency.
    sys.modules.setdefault("wandb", types.ModuleType("wandb"))
    sys.path.insert(0, str(grace_repo))
    src = grace_repo / "grace" / "editors" / "grace.py"
    spec = importlib.util.spec_from_file_location("grace_official_editor", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GRACE


class WrappedModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


def tokenize_one(tokenizer, example, device):
    enc = tokenizer(
        [example["text"]],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    enc["labels"] = torch.tensor([int(example["label"])], dtype=torch.long)
    return {k: v.to(device) for k, v in enc.items()}


def tokenize_many(tokenizer, examples, device):
    enc = tokenizer(
        [e["text"] for e in examples],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    labels = torch.tensor([int(e["label"]) for e in examples], dtype=torch.long)
    enc["labels"] = labels
    return {k: v.to(device) for k, v in enc.items()}


def get_adapter(editor):
    return editor.model.bert.encoder.layer[10].output.dense


def snapshot_memory(adapter):
    return {
        "keys": adapter.keys.detach().clone(),
        "values": adapter.values.detach().clone(),
        "epsilons": adapter.epsilons.detach().clone(),
        "key_labels": copy.deepcopy(adapter.key_labels),
    }


def restore_memory(adapter, snap, values=None):
    adapter.keys = snap["keys"].detach().clone().to(adapter.device)
    adapter.epsilons = snap["epsilons"].detach().clone().to(adapter.device)
    adapter.key_labels = copy.deepcopy(snap["key_labels"])
    v = snap["values"] if values is None else values
    adapter.values = torch.nn.Parameter(v.detach().clone().to(adapter.device), requires_grad=True)
    adapter.training = False


def capture_query(editor, tokens):
    adapter = get_adapter(editor)
    box = {}

    def hook(module, args):
        hidden = args[0]
        token_to_edit = min(adapter.key_id, hidden.shape[1] - 1)
        box["q"] = hidden[:, token_to_edit, :].detach().clone()

    h = adapter.register_forward_pre_hook(hook)
    with torch.no_grad():
        editor.model(**tokens)
    h.remove()
    return box["q"]


def route_query(adapter, query):
    d = torch.cdist(adapter.keys, query, p=2).view(-1)
    dist, idx = d.min(0)
    eps = adapter.epsilons[idx]
    return int(idx.item()), float(dist.item()), float(eps.item()), bool(dist <= eps)


def grad_values(editor, tokens):
    adapter = get_adapter(editor)
    editor.model.zero_grad(set_to_none=True)
    if adapter.values.grad is not None:
        adapter.values.grad = None
    out = editor.model(**tokens)
    out.loss.backward()
    g = adapter.values.grad.detach().clone()
    editor.model.zero_grad(set_to_none=True)
    return g


def eval_examples(editor, tokenizer, examples, device, batch_size=16):
    logits_all = []
    labels_all = []
    editor.model.eval()
    adapter = get_adapter(editor)
    adapter.training = False
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = tokenize_many(tokenizer, examples[i:i+batch_size], device)
            labels = batch["labels"]
            out = editor.model(**batch)
            logits_all.append(out.logits.detach().cpu())
            labels_all.append(labels.detach().cpu())
    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    margins = normalized_margin(logits, labels)
    acc = float((logits.argmax(dim=1) == labels).float().mean().item())
    return margins.numpy().astype(np.float64), acc


def fit_cubic(us, ys):
    x = np.asarray([u for u in us if u != 0.0], dtype=np.float64)
    y = np.stack([ys[i] for i, u in enumerate(us) if u != 0.0], axis=0)
    design = np.stack([x, x**2, x**3], axis=1)
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    pred = design @ beta
    resid = y - pred
    return beta, resid


def rank_basis(x, rel=RANK_REL):
    x = np.asarray(x, dtype=np.float64)
    ncols = x.shape[1] if x.ndim == 2 else TARGET_KEYS
    if x.size == 0:
        return 0, np.zeros((ncols, 0), dtype=np.float64), []
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    if len(s) == 0 or s[0] <= 0:
        return 0, np.zeros((ncols, 0), dtype=np.float64), s.tolist()
    tau = max(1e-8, rel * s[0])
    r = int(np.sum(s >= tau))
    return r, vt[:r].T, s.tolist()


def projector_distance(q1, q2):
    if q1.shape[1] != q2.shape[1]:
        return 1.0
    p1 = q1 @ q1.T
    p2 = q2 @ q2.T
    return float(np.linalg.norm(p1 - p2, ord=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grace-repo", default="GRACE")
    ap.add_argument("--out", default="results/grace_ic_dev.json")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cpu")

    grace_repo = Path(args.grace_repo)
    head = subprocess.check_output(["git", "-C", str(grace_repo), "rev-parse", "HEAD"], text=True).strip()
    if head != GRACE_COMMIT:
        raise RuntimeError(f"GRACE commit mismatch: {head}")
    GRACE = load_grace_class(grace_repo)

    api = HfApi()
    model_rev = api.model_info(MODEL_ID).sha
    tokenizer_rev = api.model_info(TOKENIZER_ID).sha
    dataset_rev = api.dataset_info(DATASET_ID).sha

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=tokenizer_rev)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, revision=model_rev).to(device)
    model.eval()

    wrapped = WrappedModel(model, tokenizer)
    config = {
        "device": str(device),
        "model": {"inner_params": [GRACE_LAYER]},
        "experiment": {"task": "scotus"},
        "editor": {
            "n_iter": 100,
            "edit_lr": 1.0,
            "eps": 1.0,
            "dist_fn": "euc",
            "val_init": "cold",
            "val_train": "sgd",
            "val_reg": None,
            "reg": "early_stop",
            "replacement": "replace_prompt",
            "eps_expand": "coverage",
            "num_pert": 8,
        },
    }
    editor = GRACE(config, wrapped)
    adapter = get_adapter(editor)

    ds = load_dataset(DATASET_ID, split="test", revision=dataset_rev)
    examples = [{"text": ds[i]["text"], "label": int(ds[i]["label"]), "idx": i} for i in range(len(ds))]
    examples.sort(key=stable_key)
    need = ANCHOR_CANDIDATES + PROBE_CANDIDATES + DEV_CANDIDATES + HELDOUT_CANDIDATES
    if len(examples) < need:
        raise RuntimeError(f"dataset too small: need {need}, got {len(examples)}")

    anchor_pool = examples[:ANCHOR_CANDIDATES]
    probe_pool = examples[ANCHOR_CANDIDATES:ANCHOR_CANDIDATES+PROBE_CANDIDATES]
    dev_pool = examples[ANCHOR_CANDIDATES+PROBE_CANDIDATES:ANCHOR_CANDIDATES+PROBE_CANDIDATES+DEV_CANDIDATES]
    heldout_pool = examples[ANCHOR_CANDIDATES+PROBE_CANDIDATES+DEV_CANDIDATES:need]

    role_hashes = {
        "anchor_pool": sha_text("\n".join(stable_key(x) for x in anchor_pool)),
        "probe_pool": sha_text("\n".join(stable_key(x) for x in probe_pool)),
        "dev_pool": sha_text("\n".join(stable_key(x) for x in dev_pool)),
        "heldout_pool": sha_text("\n".join(stable_key(x) for x in heldout_pool)),
    }

    creator_by_slot = {}
    successful_edits = []
    for ex in anchor_pool:
        tokens = tokenize_one(tokenizer, ex, device)
        with torch.no_grad():
            pred = int(editor.model(**tokens).logits.argmax(dim=1).item())
        if pred == ex["label"]:
            continue
        before = len(adapter.keys) if hasattr(adapter, "keys") else 0
        editor.edit(config, tokens, batch_history=[])
        after = len(adapter.keys)
        successful_edits.append({"idx": ex["idx"], "hash": stable_key(ex), "before_keys": before, "after_keys": after})
        if after > before:
            for slot in range(before, after):
                creator_by_slot[slot] = ex
        if after >= TARGET_KEYS:
            break
        if len(successful_edits) >= MAX_ANCHOR_EDITS:
            break

    if not hasattr(adapter, "keys") or len(adapter.keys) < TARGET_KEYS:
        status = {
            "status": "ANCHOR_TOPOLOGY_DEGENERATE",
            "heldout_evaluated": False,
            "nkeys": int(len(adapter.keys)) if hasattr(adapter, "keys") else 0,
            "successful_edits": successful_edits,
            "role_hashes": role_hashes,
            "model_revision": model_rev,
            "dataset_revision": dataset_rev,
            "grace_commit": GRACE_COMMIT,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(status, indent=2))
        print(json.dumps(status, indent=2))
        return

    full_snap = snapshot_memory(adapter)
    snap = {
        "keys": full_snap["keys"][:TARGET_KEYS].detach().clone(),
        "values": full_snap["values"][:TARGET_KEYS].detach().clone(),
        "epsilons": full_snap["epsilons"][:TARGET_KEYS].detach().clone(),
        "key_labels": full_snap["key_labels"][:TARGET_KEYS],
    }
    restore_memory(adapter, snap)
    base_values = snap["values"].detach().clone()

    slot_dirs = []
    state_creator_audit = []
    for slot in range(TARGET_KEYS):
        ex = creator_by_slot.get(slot)
        if ex is None:
            raise RuntimeError(f"missing creator for slot {slot}")
        restore_memory(adapter, snap)
        tokens = tokenize_one(tokenizer, ex, device)
        q = capture_query(editor, tokens)
        ridx, dist, eps, inside = route_query(adapter, q)
        g = grad_values(editor, tokens)
        if (not inside) or ridx != slot:
            raise RuntimeError(f"creator route drift slot={slot} routed={ridx} inside={inside}")
        gs = torch.zeros_like(g)
        gs[slot] = g[slot]
        norm = float(gs.norm().item())
        if norm < 1e-12:
            raise RuntimeError(f"zero creator gradient slot {slot}")
        slot_dirs.append((-gs / norm).detach())
        state_creator_audit.append({"slot": slot, "idx": ex["idx"], "hash": stable_key(ex), "grad_norm": norm, "route_dist": dist, "epsilon": eps})

    H = hadamard4(device, base_values.dtype)
    chart_dirs = []
    for j in range(TARGET_KEYS):
        d = torch.zeros_like(base_values)
        for slot in range(TARGET_KEYS):
            d = d + H[j, slot] * slot_dirs[slot]
        d = d / d.norm()
        chart_dirs.append(d.detach())
    gram = torch.stack([d.flatten() for d in chart_dirs]) @ torch.stack([d.flatten() for d in chart_dirs]).T
    chart_orth_error = float((gram - torch.eye(TARGET_KEYS)).abs().max().item())
    if chart_orth_error > 1e-5:
        raise RuntimeError(f"chart not orthonormal: {chart_orth_error}")

    probe_dirs = {}
    probe_examples = {}
    probe_audit = {}
    for ex in probe_pool:
        if len(probe_dirs) >= PROBE_COUNT:
            break
        restore_memory(adapter, snap)
        tokens = tokenize_one(tokenizer, ex, device)
        q = capture_query(editor, tokens)
        ridx, dist, eps, inside = route_query(adapter, q)
        if (not inside) or ridx in probe_dirs or ridx >= TARGET_KEYS:
            continue
        g = grad_values(editor, tokens)
        gs = torch.zeros_like(g)
        gs[ridx] = g[ridx]
        norm = float(gs.norm().item())
        if norm < 1e-12:
            continue
        probe_dirs[ridx] = (-gs / norm).detach()
        probe_examples[ridx] = ex
        probe_audit[ridx] = {"idx": ex["idx"], "hash": stable_key(ex), "grad_norm": norm, "route_dist": dist, "epsilon": eps}

    if len(probe_dirs) < PROBE_COUNT:
        status = {
            "status": "PROBE_COVERAGE_DEGENERATE",
            "heldout_evaluated": False,
            "covered_slots": sorted(probe_dirs),
            "role_hashes": role_hashes,
            "model_revision": model_rev,
            "dataset_revision": dataset_rev,
            "grace_commit": GRACE_COMMIT,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(status, indent=2))
        print(json.dumps(status, indent=2))
        return

    probe_slots = tuple(sorted(probe_dirs))

    dev_examples = []
    dev_routes = []
    for ex in dev_pool:
        restore_memory(adapter, snap)
        tokens = tokenize_one(tokenizer, ex, device)
        q = capture_query(editor, tokens)
        ridx, dist, eps, inside = route_query(adapter, q)
        if inside and ridx < TARGET_KEYS:
            dev_examples.append(ex)
            dev_routes.append({"idx": ex["idx"], "slot": ridx, "dist": dist, "epsilon": eps, "hash": stable_key(ex)})
            if len(dev_examples) >= DEV_CONSEQUENCE_N:
                break
    if len(dev_examples) < DEV_CONSEQUENCE_N:
        status = {
            "status": "DEV_SUPPORT_DEGENERATE",
            "heldout_evaluated": False,
            "dev_supported_n": len(dev_examples),
            "role_hashes": role_hashes,
            "model_revision": model_rev,
            "dataset_revision": dataset_rev,
            "grace_commit": GRACE_COMMIT,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(status, indent=2))
        print(json.dumps(status, indent=2))
        return

    eval_cache = {}
    def evaluate_values(values):
        key = tensor_sha(values)
        if key not in eval_cache:
            restore_memory(adapter, snap, values=values)
            eval_cache[key] = eval_examples(editor, tokenizer, dev_examples, device, batch_size=16)
        return eval_cache[key]

    base_margin, base_acc = evaluate_values(base_values)

    eta_report = []
    passing = []
    for eta in ETA_GRID:
        eta_rec = {"eta": eta, "probes": {}, "all_pass": True}
        for p in probe_slots:
            ys = []
            for u in U_GRID:
                vals = base_values + float(u * eta) * probe_dirs[p]
                m, _ = evaluate_values(vals)
                ys.append(m - base_margin)
            beta, resid = fit_cubic(U_GRID, ys)
            pos = np.stack([ys[0], ys[-1]], axis=0)
            signal = float(np.sqrt(np.mean(pos**2)))
            nrmse = float(np.sqrt(np.mean(resid**2)) / max(signal, 1e-12))
            max_abs = float(np.max(np.abs(np.stack(ys, axis=0))))
            ok = signal >= SIGNAL_RMS_MIN and nrmse <= CURVE_NRMSE_MAX and max_abs <= MAX_ABS_RESPONSE
            eta_rec["probes"][str(p)] = {"signal_RMS": signal, "curve_NRMSE": nrmse, "max_abs_response": max_abs, "pass": ok}
            eta_rec["all_pass"] = eta_rec["all_pass"] and ok
        eta_report.append(eta_rec)
        if eta_rec["all_pass"]:
            passing.append(eta)

    if not passing:
        status = {
            "status": "NO_LEGAL_LOW_ORDER_WINDOW",
            "heldout_evaluated": False,
            "eta_grid": eta_report,
            "base_accuracy": base_acc,
            "role_hashes": role_hashes,
            "model_revision": model_rev,
            "dataset_revision": dataset_rev,
            "grace_commit": GRACE_COMMIT,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(status, indent=2))
        print(json.dumps(status, indent=2))
        return

    eta = max(passing)
    epsilon = eta

    coeff = {j: {} for j in range(TARGET_KEYS)}
    raw_state_acc = {}
    for j in range(TARGET_KEYS):
        coeff[j] = {}
        raw_state_acc[str(j)] = {}
        for sign in (-1, 1):
            state_values = base_values + float(sign * epsilon) * chart_dirs[j]
            state_margin, state_acc = evaluate_values(state_values)
            raw_state_acc[str(j)][str(sign)] = state_acc
            coeff[j][sign] = {}
            for p in probe_slots:
                ys = []
                for u in U_GRID:
                    vals = state_values + float(u * eta) * probe_dirs[p]
                    m, _ = evaluate_values(vals)
                    ys.append(m - state_margin)
                beta, _ = fit_cubic(U_GRID, ys)
                coeff[j][sign][p] = beta

    def xi_for(k, subset):
        rows = []
        for p in subset:
            for order in range(k):
                for c in range(DEV_CONSEQUENCE_N):
                    row = []
                    for j in range(TARGET_KEYS):
                        plus = coeff[j][1][p][order, c]
                        minus = coeff[j][-1][p][order, c]
                        row.append((plus - minus) / 2.0)
                    rows.append(row)
        if not rows:
            return np.zeros((0, TARGET_KEYS), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)

    subsets = []
    for r in range(len(probe_slots) + 1):
        subsets.extend(itertools.combinations(probe_slots, r))

    cache = {}
    full_report = {}
    rank_diag = {}
    subset_report = {}
    all_tuple = tuple(probe_slots)
    for k in ORDERS:
        subset_report[str(k)] = {}
        for P in subsets:
            x = xi_for(k, P)
            r, q, s = rank_basis(x, RANK_REL)
            cache[(k, P)] = (r, q, s)
            subset_report[str(k)][",".join(map(str, P))] = {"rank": r, "singular_values": s}
        fr, fq, fs = cache[(k, all_tuple)]
        full_report[str(k)] = {"rank": fr, "singular_values": fs}
        rank_diag[str(k)] = {}
        xfull = xi_for(k, all_tuple)
        for rel in RANK_DIAG:
            rr, _, ss = rank_basis(xfull, rel)
            rank_diag[str(k)][str(rel)] = {"rank": rr, "singular_values": ss}

    r_star, q_star, s_star = cache[(3, all_tuple)]
    qk = {}
    examples_min = {}
    equality = {}
    for k in ORDERS:
        good = []
        equality[str(k)] = {}
        for P in subsets:
            r, q, s = cache[(k, P)]
            dist = projector_distance(q, q_star) if r == r_star else 1.0
            eq = (r == r_star and dist <= PROJECTOR_TOL)
            equality[str(k)][",".join(map(str, P))] = {"equal_target": eq, "projector_distance": dist, "rank": r}
            if eq:
                good.append(P)
        if good:
            min_size = min(len(P) for P in good)
            mins = sorted([P for P in good if len(P) == min_size])
            qk[str(k)] = min_size
            examples_min[str(k)] = list(mins[0])
        else:
            qk[str(k)] = None
            examples_min[str(k)] = None

    r1 = full_report["1"]["rank"]
    r2 = full_report["2"]["rank"]
    r3 = full_report["3"]["rank"]
    delayed = (r2 > r1) or (r3 > r2)
    q1 = qk["1"] if qk["1"] is not None else math.inf
    q2 = qk["2"] if qk["2"] is not None else math.inf
    q3 = qk["3"] if qk["3"] is not None else math.inf
    substitution = (q2 < q1) or (q3 < q2)
    breadth_nontrivial = q1 > 1
    labels = []
    if delayed:
        labels.append("DELAYED_HIGHER_ORDER_VISIBILITY")
    if substitution:
        labels.append("BREADTH_DEPTH_SUBSTITUTION")
    if breadth_nontrivial:
        labels.append("NONTRIVIAL_FIRST_ORDER_BREADTH")
    if q1 == 1 and not delayed:
        labels.append("FIRST_ORDER_SINGLE_PROBE_CLOSURE")
    disposition = "NONTRIVIAL_DEV_FRONTIER" if (delayed or substitution or breadth_nontrivial) else "DEV_COLLAPSE_11"

    result = {
        "status": "GRACE_IC_DEV_COMPLETE",
        "disposition": disposition,
        "scientific_labels": labels,
        "heldout_evaluated": False,
        "carrier": "GRACE BERT/SCOTUS fixed-topology value-state local write geometry",
        "grace_commit": GRACE_COMMIT,
        "model_id": MODEL_ID,
        "model_revision": model_rev,
        "dataset_id": DATASET_ID,
        "dataset_revision": dataset_rev,
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_revision": tokenizer_rev,
        "seed": SEED,
        "target_keys": TARGET_KEYS,
        "anchor_successful_edits": successful_edits,
        "anchor_memory": {
            "nkeys": TARGET_KEYS,
            "keys_sha256": tensor_sha(snap["keys"]),
            "values_sha256": tensor_sha(base_values),
            "epsilons": snap["epsilons"].detach().cpu().view(-1).tolist(),
        },
        "state_creator_audit": state_creator_audit,
        "chart_orthogonality_max_error": chart_orth_error,
        "probe_audit": {str(k): v for k, v in probe_audit.items()},
        "probe_slots": list(probe_slots),
        "dev_consequence_n": len(dev_examples),
        "dev_consequence_hash": sha_text("\n".join(stable_key(x) for x in dev_examples)),
        "dev_routes": dev_routes,
        "heldout_pool_hash": role_hashes["heldout_pool"],
        "role_hashes": role_hashes,
        "base_dev_accuracy": base_acc,
        "eta_grid": eta_report,
        "selected_eta": eta,
        "state_epsilon": epsilon,
        "full_rank_spectrum": full_report,
        "rank_diagnostics": rank_diag,
        "r_star": r_star,
        "q_k": qk,
        "minimum_example_subset": examples_min,
        "delayed_higher_order_visibility": delayed,
        "breadth_depth_substitution": substitution,
        "nontrivial_first_order_breadth": breadth_nontrivial,
        "subset_target_equality": equality,
        "subset_rank_report": subset_report,
        "raw_state_accuracy": raw_state_acc,
        "evaluation_cache_entries": len(eval_cache),
        "protocol_thresholds": {
            "rank_rel": RANK_REL,
            "rank_diag": list(RANK_DIAG),
            "projector_tol": PROJECTOR_TOL,
            "signal_rms_min": SIGNAL_RMS_MIN,
            "curve_nrmse_max": CURVE_NRMSE_MAX,
            "max_abs_response": MAX_ABS_RESPONSE,
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "status": result["status"],
        "disposition": result["disposition"],
        "scientific_labels": result["scientific_labels"],
        "selected_eta": eta,
        "full_rank_spectrum": full_report,
        "q_k": qk,
        "r_star": r_star,
        "eval_cache_entries": len(eval_cache),
    }, indent=2))


if __name__ == "__main__":
    main()
