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
    "agents-approved-persistence": ("src/agents/run_internal/run_loop.py", "openai-approved-tool-persistence"),
    "autogen-tool-cancellation": ("python/packages/autogen-agentchat/src/autogen_agentchat/agents/_assistant_agent.py", "autogen-cancellation-stream"),
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
                            "eligible":True,  # adapter supports both synchronous and async runner boundaries
                        })
    return {"search_space":{"top_level_function_count":len(funcs),
                            "tracked_futures_id_comparisons":comparisons},
            "candidates":sorted(proposals,key=lambda x:(x["is_async"],x["source_line"]))}


def _scan_approved_persistence(tree, relpath):
    """Find a resume-owned session save that occurs after a throwing guardrail.

    This recognizes callable/session relations in the SDK source, not a
    particular issue number or expected version. It is only a risk witness:
    the actual approved side effect and durable session must be run natively.
    """
    funcs=_functions(tree)
    found=[]
    for name,fn in funcs.items():
        if not _writes_final_output(fn):
            continue
        guard_indices=[i for i,stmt in enumerate(fn.body)
                       if "_run_output_guardrails_for_stream" in _called_names(stmt)]
        if not guard_indices:
            continue
        boundary_index=guard_indices[0]
        def has_save(stmt):
            return "save_items" in _called_names(stmt)
        before=any(has_save(stmt) for stmt in fn.body[:boundary_index])
        after=any(has_save(stmt) for stmt in fn.body[boundary_index+1:])
        if not after and not before:
            continue
        # Require a call site using the explicitly resumed/session-writing
        # callback, so an ordinary newly generated final answer is excluded.
        resumed_calls=[]
        for caller_name,caller in funcs.items():
            for sub in ast.walk(caller):
                if (isinstance(sub,ast.Call) and isinstance(sub.func,ast.Name)
                    and sub.func.id==name):
                    for kw in sub.keywords:
                        if (kw.arg=="save_items" and isinstance(kw.value,ast.Name)
                            and "resumed" in kw.value.id.casefold()):
                            resumed_calls.append((caller_name,sub))
        if not resumed_calls:
            continue
        protections=[]
        for caller,call in resumed_calls:
            values={kw.arg:kw.value for kw in call.keywords}
            flag=values.get("persist_before_output_guardrails")
            flag_true=isinstance(flag,ast.Constant) and flag.value is True
            protections.append(bool(before and flag_true))
        conditional_save_before_guardrail=before and all(protections)
        found.append({
            "kind":"approved_effect_before_guardrail_rejection",
            "adapter":"openai-approved-tool-persistence",
            "source_file":relpath,
            "function":name,"source_line":fn.body[boundary_index].lineno,
            "source_inferred_order":("resumed_save_before_guardrail" if conditional_save_before_guardrail
                                     else "resumed_guardrail_may_precede_save"),
            "resumed_call_sites":len(resumed_calls),
            "injection":{"callback":"output_guardrail",
                         "variants":["run_pass","streamed_pass","run_trip","streamed_trip"],
                         "requires_approval":True},
            "observation":["session_call_count","session_output_count",
                           "tool_executions","guardrail_status"],
            "negative_control":"normal_and_streamed_without_tripwire",
            "contract":"approved_executed_tool_has_paired_persisted_output",
            "eligible":len(resumed_calls)>0,
        })
    return {"search_space":{"top_level_function_count":len(funcs),
                            "resumed_persistence_functions":len(found)},
            "candidates":found}


def _scan_terminal_sentinel(tree, relpath):
    """Locate a dynamically plausible orphaned async stream completion.

    A producer awaits a potentially cancelling gather, then posts an end
    sentinel only on normal completion, while its enclosing consumer blocks
    on a queue read. This is a *potential* missing-finally witness.
    Dynamic test is required before a FAIL label is licensed.
    """
    proposals=[]
    gather_total=0
    sentinel_total=0
    for node in ast.walk(tree):
        if not isinstance(node,ast.AsyncFunctionDef):
            continue
        statements=list(node.body)
        for idx,stmt in enumerate(statements):
            calls=_called_names(stmt)
            if "gather" in calls:
                gather_total+=1
            if "put_nowait" in calls and any(
                isinstance(sub,ast.Constant) and sub.value is None for sub in ast.walk(stmt)
            ):
                sentinel_total+=1
        nested=[
            sub for sub in ast.walk(node)
            if isinstance(sub,ast.AsyncFunctionDef) and sub is not node
        ]
        for inner in nested:
            statements=list(inner.body)
            for i,stmt in enumerate(statements):
                if "gather" not in _called_names(stmt):
                    continue
                for j,after in enumerate(statements[i+1:],start=i+1):
                    if not (isinstance(after,ast.Expr) and
                            isinstance(after.value,ast.Call) and
                            isinstance(after.value.func,ast.Attribute) and
                            after.value.func.attr=="put_nowait" and
                            after.value.args and isinstance(after.value.args[0],ast.Constant)
                            and after.value.args[0].value is None):
                        continue
                    # Direct statement prefix means the sentinel does NOT live
                    # in a finally handler enclosing the potentially aborting await.
                    has_waiting_consumer=any(
                        isinstance(sub,ast.Await) and
                        isinstance(sub.value,ast.Call) and
                        isinstance(sub.value.func,ast.Attribute) and
                        sub.value.func.attr=="get"
                        for sub in ast.walk(node)
                        if sub not in ast.walk(inner)
                    )
                    task_spawn=any(
                        isinstance(sub,ast.Call) and
                        isinstance(sub.func,ast.Attribute) and
                        sub.func.attr=="create_task" and
                        any(isinstance(arg,ast.Call) and isinstance(arg.func,ast.Name)
                                and arg.func.id==inner.name for arg in sub.args)
                        for sub in ast.walk(node)
                    )
                    if not has_waiting_consumer or not task_spawn:
                        continue
                    proposals.append({
                        "kind":"cancellation_to_missing_stream_terminator",
                        "adapter":"autogen-cancellation-stream",
                        "source_file":relpath,
                        "function":node.name,
                        "producer":inner.name,
                        "source_line":stmt.lineno,
                        "terminal_line":after.lineno,
                        "waiting_consumer":True,
                        "sentinel_in_finally":False,
                        "injection":{"callback":"tool_run", "variants":["normal_complete","cancel_inflight"]},
                        "observation":["tool_started","stream_terminal","cancel_propagated","bounded_timeout"],
                        "negative_control":"identical_tool_completes_without_cancel",
                        "contract":"inflight_cancel_must_not_deadlock_stream_consumer",
                        "eligible":True,
                    })
    return {
        "search_space":{"nested_async_producers":sum(
            1 for x in ast.walk(tree) if isinstance(x,ast.AsyncFunctionDef)
            and any(isinstance(sub,ast.AsyncFunctionDef) for sub in ast.walk(x) if sub is not x)),
            "gather_statements":gather_total,
            "none_sentinel_statements":sentinel_total},
        "candidates":sorted(proposals,key=lambda x:(x["source_file"],x["source_line"]))
    }


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
    if profile=="agents-guardrail":
        scan=_scan_guardrail(tree,rel)
    elif profile=="agents-approved-persistence":
        scan=_scan_approved_persistence(tree,rel)
    elif profile=="autogen-tool-cancellation":
        scan=_scan_terminal_sentinel(tree,rel)
    else:
        scan=_scan_admission(tree,rel)
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
    # A source hash alone does not authenticate the scenario-selection logic.
    # Re-derive the selected cases from pinned bytes, to reject tampered plans.
    selected=plan.get("selected")
    if not isinstance(selected,list) or not selected:
        raise ValueError("empty/invalid selected cases")
    regenerated=synthesize(root,profile,budget=len(selected))
    for field in ("selected","candidate_count","runnable_count","search_space"):
        if plan.get(field)!=regenerated[field]:
            raise ValueError("scenario plan differs from source-derived selection: "+field)


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
