"""Collect an ACTUAL local healthy SDK source-line trace for B3 comparison.

B3 is a *Sieve-inspired source-line policy*, not the Sieve OSDI original.
Trace only a no-network deterministic model/local tool and preserve CPU cost
and source identity. No fault injection, patch labels or external effects.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path


PROFILES={
    "autogen":("python/packages/autogen-agentchat/src/autogen_agentchat/agents/_assistant_agent.py",
               "microsoft/autogen"),
    "langgraph":("libs/langgraph/langgraph/pregel/_runner.py",
                 "langchain-ai/langgraph"),
}


def collect(root:Path, profile:str, timeout:float):
    if profile not in PROFILES or timeout<=0 or timeout>5:
        raise ValueError("unsupported profile or unsafe timeout")
    rel,repo=PROFILES[profile]
    root=root.resolve()
    source=(root/rel).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError("source escapes pinned upstream")
    file_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    baseline=None
    # Import dependencies first, so trace time measures the actual source
    # execution rather than package initialization.
    if profile=="autogen":
        from autogen_agentchat.agents import AssistantAgent  # noqa: F401
        from autogen_ext.models.replay import ReplayChatCompletionClient  # noqa: F401
        from execute_autogen_cancel_plan import _run_variant
        def run():
            return asyncio.run(_run_variant(False,timeout))
        def valid(v):
            return (v["terminal"]=="returned" and v["tool_started"]
                    and "ToolCallExecutionEvent" in v["events"])
    else:
        sys.path.insert(0,str(root/"libs"/"langgraph"))
        from issue8393_langgraph_child_admission import test_case
        def run():
            return test_case(False)
        def valid(v):
            return (v["logical_id_same"] is False
                    and v["new_child_admissions"]==1
                    and v["original_pending_future"] is True)

    counts=Counter()
    def tracer(frame,event,arg):
        if event=="line":
            # Compare actual path for the pinned editable checkout.
            if Path(frame.f_code.co_filename).resolve()==source:
                counts[frame.f_lineno]+=1
        return tracer
    start=time.perf_counter_ns()
    try:
        sys.settrace(tracer)
        baseline=run()
    finally:
        sys.settrace(None)
    elapsed_ms=(time.perf_counter_ns()-start)/1e6
    if not valid(baseline) or not counts:
        raise ValueError("invalid native healthy-run control / unvisited source")
    return {
        "schema":"native_healthy_source_trace_v1",
        "trace_type":"native_healthy_source_lines",
        "profile":profile,
        "upstream_source":repo,
        "source_relpath":rel,
        "source_sha256":file_sha,
        "lines":sorted(counts),
        "line_hit_counts":{str(k):counts[k] for k in sorted(counts)},
        "unique_source_lines":len(counts),
        "total_line_hits":sum(counts.values()),
        "traced_execution_ms":round(elapsed_ms,3),
        "native_healthy_observation":baseline,
        "not_a_claim":"Sieve-inspired line trace only; no original Sieve controller instrumentation",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--profile",choices=tuple(PROFILES),required=True)
    p.add_argument("--timeout",type=float,default=1.5)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    report=collect(Path(args.root),args.profile,args.timeout)
    dst=Path(args.output)
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("NATIVE_HEALTHY_SOURCE_TRACE",json.dumps({
        "profile":report["profile"],"sha":report["source_sha256"],
        "unique_lines":report["unique_source_lines"],
        "line_hits":report["total_line_hits"],
        "traced_ms":report["traced_execution_ms"],
    },sort_keys=True))


if __name__=="__main__":
    main()
