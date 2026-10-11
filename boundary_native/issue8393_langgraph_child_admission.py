"""Native LangGraph PUSH task dedup admission (issue #8393), zero external writes.

Compare pinned upstream main against an unmerged community patch branch.
The claim is only duplicate scheduling admission, NOT actual tool execution
or monetary/durable side effects. This is a regression probe, not a new method.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import weakref
from pathlib import Path


class Obj:
    pass


class Mapping(dict):
    pass


class Counter:
    def __init__(self):
        self.calls = 0
    def call_counter(self):
        self.calls += 1
        return self.calls


def test_case(same_id: bool):
    from langgraph._internal._constants import CONF, CONFIG_KEY_SCRATCHPAD
    from langgraph.pregel._runner import _call

    parent = Obj()
    parent.id = "parent-A"
    parent.config = {CONF: {CONFIG_KEY_SCRATCHPAD: Counter()}}

    old_child = Obj()
    old_child.id = "child-A"
    old_child.writes = []
    retry_child = Obj()
    retry_child.id = "child-A" if same_id else "child-B"
    retry_child.writes = []

    outstanding = Mapping()
    old_future = concurrent.futures.Future()
    outstanding[old_future] = old_child
    admissions = []

    def scheduler(task, counter, call):
        return retry_child

    def submit(func, task, *args, **kwargs):
        admissions.append(task.id)
        return concurrent.futures.Future()

    # Keep all weak-reference targets strongly reachable for the call.
    outcome = _call(
        weakref.ref(parent),
        lambda x: x,
        None,
        futures=weakref.ref(outstanding),
        schedule_task=scheduler,
        submit=weakref.ref(submit),
    )
    return {
        "logical_id_same": same_id,
        "original_child_id": old_child.id,
        "candidate_child_id": retry_child.id,
        "new_child_admissions": len(admissions),
        "original_pending_future": not old_future.done(),
        "returned_future_done": outcome.done(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expected-phase", choices=("upstream_bug", "community_fix"), required=True)
    p.add_argument("--upstream-root", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(a.upstream_root).resolve() / "libs" / "langgraph"))
    same = test_case(True)
    different = test_case(False)
    negatives_ok = (
        same["original_pending_future"] and different["original_pending_future"]
        and different["new_child_admissions"] == 1
        and same["original_child_id"] == same["candidate_child_id"]
    )
    if not negatives_ok:
        verdict = "INVALID_NATIVE_NEGATIVE_CONTROL"
    elif a.expected_phase == "upstream_bug":
        verdict = ("EXPECTED_DUPLICATE_ADMISSION" if same["new_child_admissions"] == 1
                   else "UNEXPECTED_UPSTREAM_BEHAVIOR")
    else:
        verdict = ("EXPECTED_DEDUP_ADMISSION" if same["new_child_admissions"] == 0
                   else "UNEXPECTED_COMMUNITY_PATCH_BEHAVIOR")
    record = {
        "source": "langchain-ai/langgraph",
        "issue": "https://github.com/langchain-ai/langgraph/issues/8393",
        "patch_pr": "https://github.com/langchain-ai/langgraph/pull/8485",
        "patch_status": "closed_unmerged_community_code",
        "source_sha": a.source_commit,
        "same_id": same, "different_id_negative_control": different,
        "verdict": verdict,
        "scope": "duplicate child scheduling admission only; no external effect executed",
    }
    target=Path(a.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print("PUSH_DEDUP_NATIVE",json.dumps(record,sort_keys=True),flush=True)
    if verdict.startswith(("INVALID_", "UNEXPECTED_")):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
