"""Reproducible same-source comparison of classical adapters and our increment.

No BUG/fix labels enter candidate generation. No unsupported candidate is
counted as a successful native injection. Source micro-fixture eligibility is
NOT an observed software bug. This does not run original LDFI/Sieve/FailMapper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from classical_cut_baselines import (
    raw_sites, uniform_sites, source_local_windows,
    source_lineage_windows, trace_guided_sites,
)
from contract_delta import source_to_obligation_plans
from obligation_window_kernel import compile_source, choose_by_obligation


def time_method(fn, repeats=5):
    times=[]
    result=None
    for _ in range(repeats):
        t=time.perf_counter_ns()
        obj=fn()
        elapsed=(time.perf_counter_ns()-t)/1e6
        times.append(elapsed)
        if result is None:
            result=obj
        elif obj!=result:
            raise ValueError("non-deterministic source analysis across repeats")
    return result,round(statistics.median(times),4)


def _short_candidates(rows, cap):
    return [
        {"caller":c.get("caller"),"line":c.get(
            "line",c.get("await_at_line",c.get("guard_at_line"))),
         "kind":c.get("kind",c.get("source_obligation"))}
        for c in rows[:cap]
    ]


def matrix_for_source(source, catalog, adapters, budget, path, trace=None):
    raw,t_raw=time_method(lambda: raw_sites(source))
    b0,t0=time_method(lambda:uniform_sites(source,budget))
    b1,t1=time_method(lambda:source_local_windows(source))
    b2,t2=time_method(lambda:source_lineage_windows(source))
    b3,t3=time_method(lambda:trace_guided_sites(source,trace,budget))
    m0,t4=time_method(lambda:compile_source(source,path=path))
    m0cuts=choose_by_obligation(m0["candidate_windows"],budget)
    m1,t5=time_method(lambda:source_to_obligation_plans(
        source,catalog=catalog,supported_adapters=adapters,
        budget=budget,path=path,
    ))
    return {
        "source":path,
        "source_sha256":hashlib.sha256(source.encode()).hexdigest(),
        "budget":budget,
        "candidate_universe":{"syntactic_raw_cuts":len(raw),
                              "same_source":"ALL METHODS",
                              "whole_file_policy":"all inspect same full input"},
        "B0_uniform":{"proposed":len(b0),"selected":_short_candidates(b0,budget),
                      "median_source_ms":t0},
        "B1_local_source":{"proposed":len(b1),
                           "selected":_short_candidates(b1,budget),
                           "median_source_ms":t1},
        "B2_lineage_inspired":{"proposed":len(b2),
                               "selected":_short_candidates(b2,budget),
                               "median_source_ms":t2,
                               "not_official_Molly":True},
        "B3_trace_inspired":{"status":b3["status"],
                             "proposed":b3.get("eligible_observed"),
                             "selected":_short_candidates(b3["selected"],budget),
                             "median_source_ms":t3,
                             "not_official_Sieve":True},
        "M0_obligation_windows":{"proposed":m0["candidate_count"],
                                 "selected":_short_candidates(m0cuts,budget),
                                 "median_source_ms":t4},
        "M1_contract_delta":{"proposed":m1["source_candidate_count"],
                             "ready":m1["source_supported_count"],
                             "unknown":m1["source_unknown_count"],
                             "selected_candidate_ids":m1["selected"],
                             "estimated_native_pair_attempts":m1["estimated_native_pair_attempts"],
                             "median_source_ms":t5},
        "measurement_limit":"SOURCE-CANDIDATE COUNTS ONLY; no classical method native execution at unsupported cuts",
        "timing_limit":"median of 5 warm isolated Python calls; excludes SDK install/adapter authoring/hosted API",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--catalog",required=True)
    p.add_argument("--adapters",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--budget",type=int,default=2)
    p.add_argument("--normal-trace",default=None)
    args=p.parse_args()
    source=Path(args.source).read_text(encoding="utf-8")
    catalog=json.loads(Path(args.catalog).read_text())
    adapters=json.loads(Path(args.adapters).read_text())
    trace=None
    if args.normal_trace:
        obj=json.loads(Path(args.normal_trace).read_text())
        digest=hashlib.sha256(source.encode()).hexdigest()
        if obj.get("source_sha256")!=digest or obj.get("trace_type")!="native_healthy_source_lines":
            raise ValueError("trace source digest or provenance does not match")
        trace=set(obj["lines"])
    result=matrix_for_source(
        source,catalog,adapters,args.budget,args.source,trace,
    )
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("BASELINE_SOURCE_COMPARISON",json.dumps({
        "source_sha256":result["source_sha256"],
        "raw":result["candidate_universe"]["syntactic_raw_cuts"],
        "budget":args.budget,
        "B0":result["B0_uniform"]["proposed"],
        "B1":result["B1_local_source"]["proposed"],
        "B2":result["B2_lineage_inspired"]["proposed"],
        "B3":result["B3_trace_inspired"]["status"],
        "M0":result["M0_obligation_windows"]["proposed"],
        "M1_READY":result["M1_contract_delta"]["ready"],
        "M1_UNKNOWN":result["M1_contract_delta"]["unknown"],
    },sort_keys=True))


if __name__=="__main__":
    main()
