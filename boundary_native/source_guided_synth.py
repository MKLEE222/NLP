"""Bounded source-to-scenario synthesis for native Agent SDK failure boundaries.

Input: actual upstream Python source. Output: a provenance-bound scenario plan,
NOT an execution result. AST call reachability is an overapproximation.
Profiles declare separately trusted test adapters; no arbitrary eval or exec.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import deque
from pathlib import Path


SOURCE_PROFILES = {
    "agents-guardrail": ("src/agents/run_internal/run_loop.py", "openai-agents-sdk-guardrail"),
    "langgraph-admission": ("libs/langgraph/langgraph/pregel/_runner.py", "langgraph-push-child"),
}


def _functions(tree):
    return {node.name: node for node in getattr(tree, "body", [])
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _called_names(node):
    values=set()
    for part in ast.walk(node):
        if isinstance(part, ast.Call):
            f=part.func
            if isinstance(f, ast.Name):
                values.add(f.id)
            elif isinstance(f, ast.Attribute):
                values.add(f.attr)
    return values


def _writes_final_output(node):
    for part in ast.walk(node):
        if isinstance(part, (ast.Assign, ast.AnnAssign)):
            targets=part.targets if isinstance(part, ast.Assign) else [part.target]
            if any(isinstance(x, ast.Attribute) and x.attr=="final_output" for x in targets):
                return True
    return False


def _reachable_path(calls, sink, target):
    todo=deque([(sink, (sink,))])
    seen={sink}
    while todo:
        name, trail=todo.popleft()
        if name==target:
            return list(trail)
        for nxt in sorted(calls.get(name, ())):
            if nxt not in seen:
                seen.add(nxt)
                todo.append((nxt, trail+(nxt,)))
    return None


def _broad_handlers(fn):
    for node in ast.walk(fn):
        if not isinstance(node, (ast.Try, ast.TryStar)):
            continue
        for handler in node.handlers:
            name=handler.type.id if isinstance(handler.type, ast.Name) else None
            if name!="Exception":
                continue
            tail=handler.body[-1] if handler.body else None
            if isinstance(tail, ast.Raise):
                kind="explicit_reraise"
            elif isinstance(tail, ast.Return) and isinstance(tail.value, ast.List) and not tail.value.elts:
                kind="return_empty_list"
            elif isinstance(tail, ast.Return):
                kind="return_other"
            elif any(isinstance(x, ast.Raise) for x in ast.walk(handler)):
                kind="mixed_raise"
            else:
                kind="fallthrough_possible"
            yield handler,kind


def _scan_guardrail(tree, relpath):
    funcs=_functions(tree)
    calls={n: _called_names(f) & set(funcs) for n,f in funcs.items()}
    sinks=sorted(n for n,f in funcs.items() if _writes_final_output(f))
    baseline=sum(1 for f in funcs.values() for _ in _broad_handlers(f))
    proposals=[]
    for name, fn in funcs.items():
        # An explicitly supported adapter capability, NOT an upstream issue-ID selector.
        if "guardrail" not in name.casefold() or "run_output_guardrails" not in _called_names(fn):
            continue
        for handler, exit_kind in _broad_handlers(fn):
            trails=[v for sink in sinks if (v:=_reachable_path(calls,sink,name))]
            if not trails:
                continue
            shortest=min(trails,key=lambda x:(len(x),x))
            proposals.append({
                "kind":"exception_to_accepted_output",
                "adapter":"openai-agents-sdk-guardrail",
                "source_file":relpath,
                "function":name,
                "source_line":handler.lineno,
                "sink_call_path":shortest,
                "handler_exit":exit_kind,
                "injection":{"callback":"output_guardrail","variants":["pass","raise_RuntimeError"]},
                "observation":["caught_exception","final_output_accepted"],
                "negative_control":"pass_guardrail_accepts_final",
                "contract":"raised_guardrail_must_not_accept_final",
                "eligible":exit_kind in ("return_empty_list","explicit_reraise"),
            })
    return {"search_space":{"top_level_function_count":len(funcs),
                            "broad_exception_handler_count":baseline,
                            "final_output_writer_count":len(sinks)},
            "candidates":sorted(proposals,key=lambda x:(len(x["sink_call_path"]),x["source_line"]))}


def _scan_admission(tree, relpath):
    funcs=_functions(tree)
    proposals=[]
    comparisons=0
    for name,fn in funcs.items():
        if "schedule_task" not in _called_names(fn):
            continue
        for generator in ast.walk(fn):
            if not isinstance(generator,(ast.GeneratorExp,ast.ListComp)):
                continue
            for comp in generator.generators:
                # Only search a tracked pending-futures collection.
                iter_text=ast.unparse(comp.iter)
                if "futures" not in iter_text or ".items()" not in iter_text:
                    continue
                if not isinstance(comp.target,ast.Tuple) or len(comp.target.elts)!=2:
                    continue
                value=comp.target.elts[1]
                if not isinstance(value,ast.Name):
                    continue
                for guard in comp.ifs:
                    for test in ast.walk(guard):
                        if not isinstance(test,ast.Compare) or len(test.ops)!=1 or not isinstance(test.ops[0],ast.Eq):
                            continue
                        if len(test.comparators)!=1:
                            continue
                        rhs=test.comparators[0]
                        if not isinstance(rhs,ast.Attribute) or rhs.attr!="id":
                            continue
                        lhs=test.left
                        if isinstance(lhs,ast.Name) and lhs.id==value.id:
                            cmp="stored_object_vs_candidate_id"
                        elif (isinstance(lhs,ast.Attribute) and lhs.attr=="id"
                              and isinstance(lhs.value,ast.Name) and lhs.value.id==value.id):
                            cmp="stored_id_vs_candidate_id"
                        else:
                            continue
                        comparisons+=1
                        proposals.append({
                            "kind":"inflight_logical_id_admission",
                            "adapter":"langgraph-push-child",
                            "source_file":relpath,
                            "function":name,
                            "source_line":test.lineno,
                            "is_async":isinstance(fn,ast.AsyncFunctionDef),
                            "comparison":cmp,
                            "injection":{"pending_child":True,
                                         "variants":["same_logical_id","different_logical_id"]},
                            "observation":["new_child_admissions"],
                            "negative_control":"different_id_remains_admissible",
                            "contract":"same_inflight_logical_id_not_readmitted",
                            "eligible":not isinstance(fn,ast.AsyncFunctionDef),
                        })
    return {"search_space":{"top_level_function_count":len(funcs),
                            "tracked_futures_id_comparisons":comparisons},
            "candidates":sorted(proposals,key=lambda x:(x["is_async"],x["source_line"]))}


def synthesize(root:Path,profile:str,budget:int=1):
    if profile not in SOURCE_PROFILES or budget<1:
        raise ValueError("unsupported profile or nonpositive execution budget")
    rel, adapter=SOURCE_PROFILES[profile]
    base=root.resolve()
    source=(base/rel).resolve()
    if not source.is_relative_to(base) or not source.is_file():
        raise ValueError("input source not in pinned repository")
    raw=source.read_bytes()
    tree=ast.parse(raw.decode("utf-8"),filename=rel)
    scan=_scan_guardrail(tree,rel) if profile=="agents-guardrail" else _scan_admission(tree,rel)
    chosen=[item for item in scan["candidates"] if item["eligible"]][:budget]
    return {
        "schema_version":1,"profile":profile,"adapter":adapter,
        "source_file":rel,"source_sha256":hashlib.sha256(raw).hexdigest(),
        "search_space":scan["search_space"],
        "candidate_count":len(scan["candidates"]),
        "runnable_count":sum(bool(x["eligible"]) for x in scan["candidates"]),
        "selected":chosen,"unselected_count":len(scan["candidates"])-len(chosen),
        "status":"SCENARIO_PLAN_ONLY_NOT_EXECUTION",
        "soundness_limit":"AST local call reachability is not dynamic control-flow proof",
        "rights_limit":"Supported adapter and normative contract supplied externally",
    }


def verify_pinned_source(root:Path,plan:dict):
    profile=plan.get("profile")
    if plan.get("schema_version")!=1 or profile not in SOURCE_PROFILES:
        raise ValueError("unsupported plan")
    rel, adapter=SOURCE_PROFILES[profile]
    if plan.get("source_file")!=rel or plan.get("adapter")!=adapter:
        raise ValueError("inconsistent source/adapter")
    base=root.resolve()
    source=(base/rel).resolve()
    if not source.is_relative_to(base) or not source.is_file():
        raise ValueError("source escapes repository")
    if hashlib.sha256(source.read_bytes()).hexdigest()!=plan.get("source_sha256"):
        raise ValueError("source has drifted after planning")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--upstream-root",required=True)
    p.add_argument("--profile",required=True,choices=tuple(SOURCE_PROFILES))
    p.add_argument("--output",required=True)
    p.add_argument("--budget",type=int,default=1)
    args=p.parse_args()
    result=synthesize(Path(args.upstream_root),args.profile,args.budget)
    dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("SOURCE_SCENARIOS",json.dumps(result,sort_keys=True),flush=True)
    if not result["selected"]:
        raise SystemExit("NO_ELIGIBLE_SOURCE_CANDIDATE")


if __name__=="__main__":
    main()
