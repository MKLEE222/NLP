"""Public QAEdit study allocation only; NEVER runs a model or touches private results."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SEED = "qaedit-future-collision-v1"
# Half-open bucket ranges over SHA256(group_id | SEED) modulo 10000.
BUCKETS = [
    (4600, "hist_dev"), (6600, "next_dev"), (7200, "sentinel_dev"),
    (8600, "hist_confirm"), (9600, "next_confirm"), (10000, "sentinel_confirm"),
]
SCHEDULES = {
    "dev": dict(pairs=4, length=16, checkpoints=(8, 16), future_each=4),
    "confirm": dict(pairs=16, length=32, checkpoints=(8, 16, 32), future_each=4),
}


def digest(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def norm(s):
    return " ".join(str(s).casefold().split())


def load_rows(path):
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8-sig")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            rows = parsed
        elif isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
            rows = parsed["data"]
        else:
            raise ValueError("Expected a JSON array, {'data': [...]}, or JSONL")
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise ValueError("Dataset should contain a list of records")
    return rows, hashlib.sha256(raw).hexdigest()


def clean_record(r):
    if not isinstance(r, dict):
        return None
    v = {k: r.get(k) for k in ("prompt", "target", "subject", "rephrase", "locality")}
    v["source"] = r.get("Source", r.get("source", ""))
    if any(not isinstance(v[k], str) or not v[k].strip()
           for k in ("prompt", "target", "subject", "rephrase", "source")):
        return None
    if not isinstance(v["locality"], list) or len(v["locality"]) != 5:
        return None
    if any(not isinstance(x, dict) or not isinstance(x.get("loc"), str)
           or not x["loc"].strip() or not isinstance(x.get("loc_ans"), str)
           or not x["loc_ans"].strip() for x in v["locality"]):
        return None
    v["subject_key"] = norm(v["subject"])
    v["prompt_key"] = norm(v["prompt"])
    v["rephrase_key"] = norm(v["rephrase"])
    v["uid"] = digest(json.dumps(
        [v[k] for k in ("source", "prompt", "target", "subject", "rephrase")],
        ensure_ascii=False, separators=(",", ":")))[:20]
    return v


class DSU:
    def __init__(self):
        self.parent = {}
    def find(self, key):
        if key not in self.parent:
            self.parent[key] = key
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def allocate(rows):
    invalid = 0
    duplicates = 0
    clean = {}
    for row in rows:
        r = clean_record(row)
        if r is None:
            invalid += 1
        elif r["uid"] in clean:
            duplicates += 1
        else:
            clean[r["uid"]] = r
    uf, owner = DSU(), {}
    # Connect subjects whose questions or paraphrases have identical normalized text.
    # Exact text groups remain wholly within one role, even if subject extraction differs.
    for r in clean.values():
        sub = r["subject_key"]
        uf.find(sub)
        for key in (r["prompt_key"], r["rephrase_key"]):
            if key in owner:
                uf.union(sub, owner[key])
            else:
                owner[key] = sub
    components = defaultdict(set)
    for sub in uf.parent:
        components[uf.find(sub)].add(sub)
    group_role = {}
    for root, subjects in components.items():
        group_token = "|".join(sorted(subjects))
        index = int(digest(SEED + "|group|" + group_token)[:12], 16) % 10000
        group_role[root] = next(name for boundary, name in BUCKETS
                                if index < boundary)
    pools = defaultdict(list)
    text_roles = defaultdict(set)
    for r in clean.values():
        role = group_role[uf.find(r["subject_key"])]
        pools[role].append(r["uid"])
        for text in (r["prompt_key"], r["rephrase_key"]):
            text_roles[text].add(role)
    collisions = sum(len(v) > 1 for v in text_roles.values())
    if collisions:
        raise AssertionError("Cross-role QA/paraphrase leakage: " + str(collisions))
    summary = {
        "raw_rows": len(rows), "valid_unique_rows": len(clean),
        "invalid_rows": invalid, "duplicate_rows": duplicates,
        "subject_connected_components": len(components),
        "rows_by_role": {role: len(pools.get(role, [])) for _, role in BUCKETS},
        "source_counts": dict(Counter(r["source"] for r in clean.values())),
        "cross_role_prompt_or_rephrase_leakage": collisions,
    }
    return pools, summary


def select(pools, role, count):
    ranked = sorted(pools.get(role, []),
                    key=lambda uid: (digest(SEED + "|selection|" + role + "|" + uid), uid))
    if len(ranked) < count:
        raise ValueError("Insufficient " + role + ": need " + str(count)
                         + ", available " + str(len(ranked)))
    return ranked[:count]


def phase_schedule(pools, phase):
    cfg = SCHEDULES[phase]
    num_hist = cfg["pairs"] * 2
    history = select(pools, "hist_" + phase, num_hist * cfg["length"])
    num_futures = cfg["pairs"] * len(cfg["checkpoints"]) * cfg["future_each"]
    futures = select(pools, "next_" + phase, num_futures)
    sentinels = select(pools, "sentinel_" + phase, 8)
    trajectories = []
    events = []
    index = 0
    for pair in range(cfg["pairs"]):
        left = history[(2 * pair) * cfg["length"]:(2 * pair + 1) * cfg["length"]]
        right = history[(2 * pair + 1) * cfg["length"]:(2 * pair + 2) * cfg["length"]]
        trajectories.append({"pair": pair, "A": left, "B": right})
        for checkpoint in cfg["checkpoints"]:
            for _ in range(cfg["future_each"]):
                events.append({
                    "pair": pair, "checkpoint": checkpoint,
                    "future_action_id": futures[index],
                })
                index += 1
    assert index == len(futures)
    assert len(set(history + futures + sentinels)) == len(history + futures + sentinels)
    return {
        "phase": phase, "configuration": cfg,
        "sentinel_question_ids": sentinels,
        "trajectories": trajectories, "events": events,
        "candidate_state_pair_events": len(events),
    }


def build(rows, source_sha256):
    pools, summary = allocate(rows)
    return {
        "protocol": SEED,
        "dataset": "WenDingY/QAEdit", "raw_file_sha256": source_sha256,
        "status": "PREPARED_UNRUN_NO_MODEL_OUTCOMES",
        "summary": summary,
        "dev": phase_schedule(pools, "dev"),
        "confirm": phase_schedule(pools, "confirm"),
        "claim_guard": ("Manifest contains hashed IDs and deterministic plans only. "
                        "It certifies neither model edits nor collision outcomes."),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Local QAEdit JSON array or JSONL")
    p.add_argument("--output", required=True, help="Manifest JSON, no text answers")
    args = p.parse_args()
    rows, sha = load_rows(args.input)
    manifest = build(rows, sha)
    Path(args.output).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    print("Prepared only; no model execution or confirmation outcomes.")


if __name__ == "__main__":
    main()
