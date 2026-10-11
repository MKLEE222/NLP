"""Evaluate the possible-world receipt fiber on previously run NATIVE SDK logs.

No issue-specific branching in infer_worlds. Native data is a frozen, exact
Actions-log-derived developer receipt; no fetch/heldout/remote side effect.
A local tool execution count attests *local* execution only, NOT external
payment/write commit, and cannot justify a production retry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from epistemic_receipt_fibers import (
    Receipt,infer_worlds,naive_log_only_retry,next_read_only_receipt_query,
)


def run(archive):
    if archive.get("schema")!="known_development_native_approved_tool_receipts_v1":
        raise ValueError("untrusted native log envelope")
    outcome=[]
    for item in archive["cases"]:
        source=item["exact_native_local_case"]
        if source["mode"]!="streamed" or source["tripwire"] is not True:
            raise ValueError("not the preserved streamed tripwire unit")
        if item["source_commit"] not in (
            "6e2095d918a3875ba367ccbd9cc03ecb70ce2986",
            "9d894a9032e8d373f8b54d81591db36ce75b315d",
        ):
            raise ValueError("unsupported SDK historical source")
        phase=item["phase"]
        if phase not in ("before","after"):
            raise ValueError("unknown comparison phase")
        ids=source["session_call_ids"]
        if len(ids)!=1 or len(set(ids))!=1:
            raise ValueError("missing unique call identity")
        if any(x!=ids[0] for x in source["session_output_ids"]):
            raise ValueError("mismatched output identity: separate fault case required")
        identity=phase+"@"+item["source_commit"]+"::call_"+ids[0]
        # The phase+commit qualifier is part of the identity, preventing
        # accidentally joining separate runs just because both use "2".
        receipts=[
            Receipt("sess-call-"+identity,identity,
                    "session_journal","call_recorded",True),
            Receipt("sess-result-"+identity,identity,
                    "session_journal","result_recorded",
                    ids[0] in source["session_output_ids"]),
        ]
        # The local deterministic function count is an independent execution
        # signal only within THIS fake tool run, not a remote commit receipt.
        if source["tool_executions"]==1:
            receipts.append(Receipt(
                "local-effect-"+identity,identity,
                "local_effect_counter","effect_committed",True,
            ))
        fiber=infer_worlds(identity,receipts)
        log_only=naive_log_only_retry(receipts,identity)
        query=next_read_only_receipt_query(
            fiber,["approval_authority","idempotency_receipt"])
        outcome.append({
            "phase":phase,"native_action_run":item["github_action_run"],
            "native_action_job":item["github_job_id"],
            "source_commit":item["source_commit"],
            "logical_call_id":identity,
            "native_session_call_ids":ids,
            "native_session_result_ids":source["session_output_ids"],
            "native_in_process_tool_executions":source["tool_executions"],
            "normalized_native_is_local_effect_only":True,
            "possible_worlds":fiber.json(),
            "naive_log_only_suggests":log_only,
            "permitted_read_only_query":query,
        })
    if len(outcome)!=2:
        raise ValueError("missing pre/post real native contrast")
    by={r["phase"]:r for r in outcome}
    if len(by["before"]["native_session_result_ids"])!=0 or (
        len(by["after"]["native_session_result_ids"])!=1
    ):
        raise ValueError("native historical / fixed result facts differ")
    if any(r["possible_worlds"]["can_replay_without_external_duplication"]!=
           "UNSAFE_OR_UNAUTHORIZED" for r in outcome):
        raise ValueError("committed local execution cannot be automatically replayed")
    return {
        "schema":"epistemic_native_dev_v1",
        "issue_units":1,
        "upstream":"OpenAI Agents, approved tool stream #4125",
        "known_historical_issue":True,
        "source_of_truth":"native GitHub Actions run 38119414591; no new SDK native execution",
        "comparison_class":"local log-only strawman vs set-valued incomplete evidence handling",
        "novelty_claim":"none: epistemic possible-world monitoring established in formal methods",
        "real_remote_commit_known":False,
        "results":outcome,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    report=run(json.loads(Path(args.input).read_text()))
    dst=Path(args.output)
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("EPISTEMIC_NATIVE_DEV",json.dumps({
        "issue_units":report["issue_units"],
        "results":[{
            "phase":c["phase"],
            "native_result_count":len(c["native_session_result_ids"]),
            "remaining_possible_worlds":c["possible_worlds"]["world_count"],
            "retry_assessment":c["possible_worlds"][
                "can_replay_without_external_duplication"],
            "naive_log_only":c["naive_log_only_suggests"],
        } for c in report["results"]],
    },sort_keys=True))


if __name__=="__main__":
    main()
