"""Frozen prompt-disjoint Name That LLM kill test. See PROTOCOL.md."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, normalize


HERE = Path(__file__).resolve().parent
SEED = 20261107
BOOT_SEED = 20261108


def load_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if Path(n).name == "train.csv"]
            if len(names) != 1:
                raise ValueError(f"Expected one train.csv in zip, found {names}")
            with z.open(names[0]) as f:
                return pd.read_csv(f)
    return pd.read_csv(path)


def style_features(texts):
    rows = []
    for raw in texts:
        s = str(raw)
        words = s.split()
        lines = s.splitlines()
        n = max(len(s), 1)
        rows.append([
            len(s), len(words), len(lines),
            np.mean([len(w) for w in words]) if words else 0,
            sum(c.isupper() for c in s) / n,
            sum(c.isdigit() for c in s) / n,
            s.count("."), s.count(","), s.count("!"), s.count("?"),
            s.count(";"), s.count(":"), s.count("\n"),
            sum(s.count(c) for c in '"“”\'‘’'),
            sum(s.count(c) for c in "-—–"),
        ])
    return np.asarray(rows, dtype=float)


def clf(seed=SEED):
    return LogisticRegression(max_iter=400, class_weight="balanced", C=2.0,
                              random_state=seed)


def by_prompt_center(x, groups):
    out = x.copy().astype(float)
    groups = np.asarray(groups)
    for g in np.unique(groups):
        ix = np.flatnonzero(groups == g)
        out[ix] -= x[ix].mean(axis=0)
    return out


def recover_groups(word_x, n):
    if n < 2:
        return np.zeros(n, dtype=int)
    k = min(n, max(1, math.ceil(n / 18)))
    if k == 1:
        return np.zeros(n, dtype=int)
    dense = normalize(word_x).toarray()
    return AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(dense)


def pair_features(a, b):
    return np.hstack((np.abs(a - b), a * b))


def make_pairs(z, y, groups, count=1500):
    rng = np.random.default_rng(SEED)
    y = np.asarray(y)
    groups = np.asarray(groups)
    by_class = {v: np.flatnonzero(y == v) for v in np.unique(y)}
    by_group = {v: np.flatnonzero(groups == v) for v in np.unique(groups)}
    pos = []
    neg = []
    tries = 0
    while len(pos) < count and tries < count * 100:
        tries += 1
        label = rng.choice(list(by_class))
        ix = rng.choice(by_class[label], 2, replace=False)
        if groups[ix[0]] != groups[ix[1]]:
            pos.append(ix)
    tries = 0
    while len(neg) < count and tries < count * 100:
        tries += 1
        group = rng.choice(list(by_group))
        pool = by_group[group]
        if len(pool) < 2:
            continue
        ix = rng.choice(pool, 2, replace=False)
        if y[ix[0]] != y[ix[1]]:
            neg.append(ix)
    if len(pos) != count or len(neg) != count:
        raise ValueError("Insufficient legal training pairs")
    pairs = np.vstack((pos, neg))
    labels = np.r_[np.ones(count), np.zeros(count)]
    return pair_features(z[pairs[:, 0]], z[pairs[:, 1]]), labels


def align_prob(model, x, classes):
    out = np.zeros((x.shape[0], len(classes)))
    raw = model.predict_proba(x)
    lookup = {v: i for i, v in enumerate(classes)}
    for j, label in enumerate(model.classes_):
        out[:, lookup[label]] = raw[:, j]
    return out


def assign_by_group(prob, groups, word_x, classes, threshold=True):
    predicted = np.asarray(classes)[prob.argmax(axis=1)].copy()
    coverage = 0
    groups = np.asarray(groups)
    for g in np.unique(groups):
        ix = np.flatnonzero(groups == g)
        if len(ix) != len(classes):
            continue
        if threshold:
            sim = cosine_similarity(word_x[ix])
            mean_sim = (sim.sum() - len(ix)) / (len(ix) * (len(ix) - 1))
            if mean_sim <= 0.25:
                continue
        row, col = linear_sum_assignment(-np.log(np.maximum(prob[ix], 1e-12)))
        predicted[ix[row]] = np.asarray(classes)[col]
        coverage += len(ix)
    return predicted, coverage


def recall_vector(y, pred, classes):
    mat = confusion_matrix(y, pred, labels=classes)
    denom = mat.sum(axis=1)
    return np.divide(np.diag(mat), denom, out=np.full(len(classes), np.nan), where=denom > 0)


def score(y, pred, classes):
    return float(np.nanmean(recall_vector(y, pred, classes)))


def bootstrap_diff(y, groups, base, cand, classes):
    rng = np.random.default_rng(BOOT_SEED)
    ug = np.unique(groups)
    pool = {g: np.flatnonzero(groups == g) for g in ug}
    diffs = np.empty(1000)
    for b in range(1000):
        sampled = rng.choice(ug, size=len(ug), replace=True)
        ix = np.concatenate([pool[g] for g in sampled])
        diffs[b] = score(y[ix], cand[ix], classes) - score(y[ix], base[ix], classes)
    return {"mean": float(diffs.mean()), "lower_1_667_pct": float(np.quantile(diffs, 0.05 / 3)),
            "upper_98_333_pct": float(np.quantile(diffs, 1 - 0.05 / 3))}


def main(path: Path):
    start = time.time()
    raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    df = load_data(path)
    required = {"id", "prompt_id", "text", "model"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
    if df.id.duplicated().any() or df[list(required)].isna().any().any():
        raise ValueError("Duplicate ID or missing required data")
    classes = np.asarray(sorted(df.model.unique()))
    groups_all = np.asarray(sorted(df.prompt_id.unique()))
    if len(classes) != 18 or len(groups_all) < 46:
        raise ValueError(f"Expected 18 classes and >45 prompt groups; got {len(classes)}, {len(groups_all)}")
    shuffled = np.random.default_rng(SEED).permutation(groups_all)
    lock_groups, dev_groups, fit_groups = shuffled[:25], shuffled[25:45], shuffled[45:]
    assert len(set(lock_groups) & set(dev_groups)) == 0
    assert len(set(lock_groups) & set(fit_groups)) == 0
    assert len(set(dev_groups) & set(fit_groups)) == 0
    parts = {k: df[df.prompt_id.isin(g)].reset_index(drop=True) for k, g in
             (("fit", fit_groups), ("dev", dev_groups), ("lockbox", lock_groups))}
    if set(parts["fit"].model) != set(classes):
        raise ValueError("Fit partition lacks one or more classes")
    texts = {k: p.text.astype(str).tolist() for k, p in parts.items()}
    char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                           max_features=250000, sublinear_tf=True, dtype=np.float32)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                           max_features=100000, sublinear_tf=True, dtype=np.float32,
                           token_pattern=r"(?u)\b\w+\b")
    xc = {"fit": char.fit_transform(texts["fit"])}
    xw = {"fit": word.fit_transform(texts["fit"])}
    for part in ("dev", "lockbox"):
        xc[part], xw[part] = char.transform(texts[part]), word.transform(texts[part])
    fs = {k: style_features(v) for k, v in texts.items()}
    scaler = StandardScaler().fit(fs["fit"])
    xf = {k: scaler.transform(v) for k, v in fs.items()}
    base_models = [clf().fit(xc["fit"], parts["fit"].model),
                   clf().fit(xw["fit"], parts["fit"].model),
                   clf().fit(xf["fit"], parts["fit"].model)]
    a0 = {k: sum(w * align_prob(m, x[k], classes) for w, m, x in
                 zip((.45, .35, .20), base_models, (xc, xw, xf)))
          for k in ("dev", "lockbox")}

    dim = min(96, xw["fit"].shape[1] - 1, xw["fit"].shape[0] - 1)
    svd = TruncatedSVD(n_components=dim, random_state=SEED).fit(xw["fit"])
    z = {k: normalize(svd.transform(v)) for k, v in xw.items()}
    pair_x, pair_y = make_pairs(z["fit"], parts["fit"].model, parts["fit"].prompt_id)
    pair_model = LogisticRegression(max_iter=400, random_state=SEED).fit(pair_x, pair_y)
    proto = np.vstack([normalize(z["fit"][parts["fit"].model == c].mean(axis=0).reshape(1, -1))[0]
                       for c in classes])
    a1 = {}
    for k in ("dev", "lockbox"):
        raw = np.column_stack([pair_model.predict_proba(pair_features(z[k], np.tile(p, (len(z[k]), 1))))[:, 1]
                               for p in proto])
        a1[k] = .75 * a0[k] + .25 * softmax(raw * 8, axis=1)

    residual = by_prompt_center(fs["fit"], parts["fit"].prompt_id)
    res_scaler = StandardScaler().fit(residual)
    res_model = clf().fit(res_scaler.transform(residual), parts["fit"].model)
    a2, oracle_a2, recovered, ari = {}, {}, {}, {}
    for k in ("dev", "lockbox"):
        recovered[k] = recover_groups(xw[k], len(parts[k]))
        ari[k] = float(adjusted_rand_score(parts[k].prompt_id, recovered[k]))
        xr = res_scaler.transform(by_prompt_center(fs[k], recovered[k]))
        a2[k] = .75 * a0[k] + .25 * align_prob(res_model, xr, classes)
        xo = res_scaler.transform(by_prompt_center(fs[k], parts[k].prompt_id))
        oracle_a2[k] = .75 * a0[k] + .25 * align_prob(res_model, xo, classes)

    preds, diagnostics = {}, {}
    for k in ("dev", "lockbox"):
        preds[k] = {n: np.asarray(classes)[p.argmax(axis=1)] for n, p in
                    (("A0", a0[k]), ("A1", a1[k]), ("A2", a2[k]))}
        preds[k]["A3"], cov = assign_by_group(a2[k], recovered[k], xw[k], classes)
        oracle3, oracle_cov = assign_by_group(oracle_a2[k], parts[k].prompt_id, xw[k], classes,
                                               threshold=False)
        diagnostics[k] = {"recovered_group_ARI": ari[k], "A3_coverage": cov / len(parts[k]),
                          "oracle_A2_BA": score(np.asarray(parts[k].model),
                                                 np.asarray(classes)[oracle_a2[k].argmax(axis=1)], classes),
                          "oracle_A3_BA": score(np.asarray(parts[k].model), oracle3, classes),
                          "oracle_A3_coverage": oracle_cov / len(parts[k])}

    yd = np.asarray(parts["dev"].model)
    boot = {n: bootstrap_diff(yd, np.asarray(parts["dev"].prompt_id), preds["dev"]["A0"],
                              preds["dev"][n], classes) for n in ("A1", "A2", "A3")}
    dev_scores = {n: score(yd, pred, classes) for n, pred in preds["dev"].items()}
    qualifying = [n for n in boot if boot[n]["lower_1_667_pct"] > 0]
    chosen = max(qualifying, key=lambda n: dev_scores[n] - dev_scores["A0"]) if qualifying else "A0"
    yl = np.asarray(parts["lockbox"].model)
    lock_scores = {n: score(yl, pred, classes) for n, pred in preds["lockbox"].items()}
    recalls = {n: {str(c): (None if np.isnan(v) else float(v)) for c, v in
                   zip(classes, recall_vector(yl, pred, classes))}
               for n, pred in preds["lockbox"].items()}
    # Reporting-only addition after the first frozen run; no model or choice changes.
    rng = np.random.default_rng(20261109)
    lg = np.asarray(parts["lockbox"].prompt_id)
    unique = np.unique(lg)
    pools = {g: np.flatnonzero(lg == g) for g in unique}
    samples = {n: [] for n in preds["lockbox"]}
    for _ in range(2000):
        ix = np.concatenate([pools[g] for g in rng.choice(unique, len(unique), replace=True)])
        for n, pred in preds["lockbox"].items():
            samples[n].append(score(yl[ix], pred[ix], classes))
    intervals = {n: {"BA_95pct": np.quantile(v, [.025, .975]).tolist(),
                     "paired_delta_A0_95pct": np.quantile(np.asarray(v) - samples["A0"], [.025, .975]).tolist()}
                 for n, v in samples.items()}
    normal = df.text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    partitions = df.prompt_id.map({g: k for k, p in parts.items() for g in p.prompt_id.unique()})
    duplicates = pd.DataFrame({"text": normal, "part": partitions})
    result = {
        "status": "COMPLETE", "protocol": "PROTOCOL.md 2026-11-07", "data_sha256": raw_hash,
        "data_file": path.name, "rows": len(df), "prompt_groups": len(groups_all),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "scipy": scipy.__version__},
        "prompt_group_size": {"min": int(df.groupby("prompt_id").size().min()),
                              "max": int(df.groupby("prompt_id").size().max())},
        "class_count": len(classes), "split_seed": SEED,
        "split": {k: {"rows": len(v), "prompt_groups": int(v.prompt_id.nunique())} for k, v in parts.items()},
        "development_balanced_accuracy": dev_scores, "development_paired_bootstrap": boot,
        "ccc_gate_choice": chosen, "lockbox_balanced_accuracy": lock_scores,
        "lockbox_chosen_delta_over_A0": lock_scores[chosen] - lock_scores["A0"],
        "lockbox_class_recall": recalls, "diagnostics": diagnostics,
        "lockbox_reporting_bootstrap": intervals,
        "normalized_duplicate_rows": int(normal.duplicated().sum()),
        "cross_partition_duplicate_texts": int((duplicates.groupby("text").part.nunique() > 1).sum()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest(),
        "seconds": round(time.time() - start, 2),
        "notes": ["True validation prompt_id used only for scoring and oracle diagnostics.",
                  "Recovered prompt grouping and A3 assignment use validation text without labels.",
                  "No leaderboard or hidden test used."],
    }
    target = HERE / "results.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "split", "development_balanced_accuracy",
                  "development_paired_bootstrap", "ccc_gate_choice", "lockbox_balanced_accuracy",
                  "lockbox_chosen_delta_over_A0", "diagnostics", "seconds")}, indent=2))
    print(f"Full results: {target}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python run_killtest.py <official train.csv or competition zip>")
    main(Path(sys.argv[1]).resolve())
