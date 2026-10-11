"""Obligation-window source analyzer (research v0; no bug verdicts).

Instead of scanning for a failure keyword, build a small provenance-backed
certificate: a spawned async producer owns a matching queue, a consumer waits
on that queue, a producer await may abort, and the end-of-stream notification
is after that await without a guaranteed finally-path discharge.

This is deliberately separate from the three issue-aware native adapters.
The scanner accepts arbitrary Python files, has no upstream issue IDs, and
produces only risk/test PLANs, never a BUG claim. Contracts and executable
native adapters remain external obligations.

Soundness limitation: intra-procedural structural abstraction; matching names
and AST nesting is not a full call graph, alias analysis or exception proof.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Window:
    source_sha256: str
    caller: str
    producer: str
    spawned_at_line: int
    await_at_line: int
    discharge_at_line: int
    queue_producer_name: str
    queue_consumer_name: str
    source_obligation: str = "stream_terminal_delivery"
    fault: str = "cancel_at_producer_await"
    normal_control: str = "complete_producer_await"
    required_runtime_check: str = "consumer_bounded_terminal_or_propagated_cancel"
    static_status: str = "TEST_PLAN_ONLY"


def _functions(tree: ast.AST):
    return [node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _names_of_reads(node: ast.AST, exclusions: set[int]):
    reads=set()
    for sub in ast.walk(node):
        if id(sub) in exclusions or not isinstance(sub, ast.Await):
            continue
        call=sub.value
        if not isinstance(call, ast.Call):
            continue
        getter=call.func
        if isinstance(getter, ast.Attribute) and getter.attr=="get" and isinstance(getter.value, ast.Name):
            reads.add(getter.value.id)
    return reads


def _match_spawn_alias(outer: ast.AST, inner: ast.AsyncFunctionDef, formal: str):
    """Return observed outer queue names, with AST-derived callsite evidence."""
    params=[arg.arg for arg in inner.args.posonlyargs + inner.args.args]
    out=[]
    for node in ast.walk(outer):
        if not isinstance(node,ast.Call):
            continue
        f=node.func
        if not (isinstance(f,ast.Attribute) and f.attr=="create_task"):
            continue
        for candidate in node.args:
            if not isinstance(candidate,ast.Call) or not isinstance(candidate.func,ast.Name):
                continue
            if candidate.func.id != inner.name:
                continue
            if formal not in params:
                # Closed-over outer queue: possible but not proof when shadowed;
                # abstain rather than invent an alias certificate.
                continue
            idx=params.index(formal)
            actual=(candidate.args[idx] if len(candidate.args)>idx else
                    next((kw.value for kw in candidate.keywords if kw.arg==formal),None))
            if isinstance(actual,ast.Name):
                out.append((actual.id,node.lineno))
    return out


def _is_none_sentinel(stmt: ast.AST):
    if not isinstance(stmt,ast.Expr) or not isinstance(stmt.value,ast.Call):
        return None
    call=stmt.value
    if not (isinstance(call.func,ast.Attribute) and
            call.func.attr in ("put_nowait","put") and
            isinstance(call.func.value,ast.Name) and
            len(call.args)==1 and
            isinstance(call.args[0],ast.Constant) and call.args[0].value is None):
        return None
    return call.func.value.id


def _consumer_reads_after_spawn_without_rebinding(outer, excluded_nodes, queue, spawn_line):
    """Syntactic single-binding certificate, fail closed on rebinding.

    Source-level names are not full alias identities. A queue variable
    rewritten after the task has been spawned might identify a *different*
    queue at the awaited read. Reject this candidate even if names coincide.
    """
    reads=[]
    assignments=[]
    for part in ast.walk(outer):
        if id(part) in excluded_nodes:
            continue
        if isinstance(part,ast.Await):
            getter=part.value
            if (isinstance(getter,ast.Call)
                and isinstance(getter.func,ast.Attribute)
                and getter.func.attr=="get"
                and isinstance(getter.func.value,ast.Name)
                and getter.func.value.id==queue):
                reads.append(part.lineno)
        if isinstance(part,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
            targets=(part.targets if isinstance(part,ast.Assign) else [part.target])
            if any(isinstance(target,ast.Name) and target.id==queue
                   for target in targets):
                assignments.append(part.lineno)
    return any(
        reader>spawn_line and
        not any(spawn_line<write<reader for write in assignments)
        for reader in reads
    )


def _awaits_of_statement(node: ast.AST):
    # Restrict to the *current block* and exclude local nested defs to avoid
    # assigning future checkpoints to the wrong producer.
    return [a for a in ast.walk(node) if isinstance(a, ast.Await)]


def _unsafe_await_to_discharges(producer: ast.AsyncFunctionDef):
    """Find potential cancellation cut before a *normal-path* sentinel.

    Model direct statement order, and try-body with an adjoining unprotected
    normal-path discharge. If a finally block discharges the matching queue,
    that queue is protected for this local await; exclude it.

    This is a may-risk witness, not sound proof of actual reachability.
    """
    windows=[]
    def walk_block(stmts, guarded: set[str]):
        for i,stmt in enumerate(stmts):
            # A direct terminal transfer stops normal sequential execution.
            # Other higher-order control flow is intentionally not assumed.
            if isinstance(stmt,(ast.Return,ast.Raise,ast.Break,ast.Continue)):
                break
            if isinstance(stmt,ast.Try):
                protected=set(guarded)
                for fin in stmt.finalbody:
                    for sub in ast.walk(fin):
                        name=_is_none_sentinel(sub)
                        if name:
                            protected.add(name)
                walk_block(stmt.body,protected)
                for handler in stmt.handlers:
                    walk_block(handler.body,guarded)
                walk_block(stmt.orelse,guarded)
                walk_block(stmt.finalbody,guarded)
                continue
            if isinstance(stmt, (ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                continue
            awaits=_awaits_of_statement(stmt)
            if not awaits:
                continue
            # Resolve an after-await direct statement (not a may-run branch).
            for later in stmts[i+1:]:
                if isinstance(later,(ast.Return,ast.Raise,ast.Break,ast.Continue)):
                    break
                formal=_is_none_sentinel(later)
                if formal is not None and formal not in guarded:
                    for await_site in awaits:
                        windows.append((await_site.lineno,later.lineno,formal))
    walk_block(producer.body,set())
    return windows


def _task_admission_windows(tree, digest):
    """Generate tests at logical-ID dedup gates, independent of SDK paths.

    This recognizes an existing collection of pending tasks and an attempted
    scheduled task compared by logical ID. Both object-vs-ID (potential risk)
    and ID-vs-ID (intended safe comparison) generate a paired same/different-ID
    *test plan*. Actual correctness requires native dispatch and contract.
    """
    windows=[]
    for fn in _functions(tree):
        for generator in ast.walk(fn):
            if not isinstance(generator,(ast.GeneratorExp,ast.ListComp)):
                continue
            for comp in generator.generators:
                # A collection of (future, task)-like entries.
                if not isinstance(comp.target,ast.Tuple) or len(comp.target.elts)!=2:
                    continue
                stored=comp.target.elts[1]
                if not isinstance(stored,ast.Name):
                    continue
                iterable=ast.unparse(comp.iter)
                if ".items()" not in iterable:
                    continue
                for guard in comp.ifs:
                    for cmp in ast.walk(guard):
                        if not isinstance(cmp,ast.Compare) or len(cmp.ops)!=1:
                            continue
                        if not isinstance(cmp.ops[0],(ast.Eq,ast.Is)):
                            continue
                        if len(cmp.comparators)!=1:
                            continue
                        rhs=cmp.comparators[0]
                        if not (isinstance(rhs,ast.Attribute) and rhs.attr=="id"):
                            continue
                        left=cmp.left
                        if isinstance(left,ast.Name) and left.id==stored.id:
                            existing_projection="object"
                        elif (isinstance(left,ast.Attribute) and left.attr=="id" and
                              isinstance(left.value,ast.Name) and left.value.id==stored.id):
                            existing_projection="id"
                        else:
                            continue
                        # Require enclosing scheduling logic, not arbitrary
                        # cache-key comparison in unrelated collection traversal.
                        calls={x.func.id if isinstance(x.func,ast.Name) else x.func.attr
                               for x in ast.walk(fn) if isinstance(x,ast.Call)
                               if isinstance(x.func,(ast.Name,ast.Attribute))}
                        if "schedule_task" not in calls:
                            continue
                        windows.append({
                            "source_sha256":digest,
                            "caller":fn.name,
                            "source_obligation":"inflight_logical_task_not_reissued",
                            "guard_at_line":cmp.lineno,
                            "stored_task_identifier":stored.id,
                            "attempted_identifier":ast.unparse(rhs),
                            "existing_projection":existing_projection,
                            "pending_collection":iterable,
                            "fault":"retry_with_same_logical_task_id",
                            "normal_control":"retry_with_distinct_logical_task_id",
                            "required_runtime_check":"count_extra_pending_id_admissions",
                            "static_status":"TEST_PLAN_ONLY",
                        })
    return sorted(
        { (w["caller"],w["guard_at_line"],w["existing_projection"]):w
         for w in windows}.values(),
        key=lambda x:(x["caller"],x["guard_at_line"],x["existing_projection"]),
    )


def compile_source(source: str, *, path: str="<source>") -> dict:
    raw=source.encode("utf-8")
    digest=hashlib.sha256(raw).hexdigest()
    tree=ast.parse(source,filename=path)
    allfuncs=_functions(tree)
    findings=[]
    considered=0
    exclusions=set()
    for f in allfuncs:
        if isinstance(f,ast.AsyncFunctionDef):
            nested=[n for n in ast.walk(f) if isinstance(n,ast.AsyncFunctionDef) and n is not f]
            exclusions={id(x) for inner in nested for x in ast.walk(inner)}
            reads=_names_of_reads(f,exclusions)
            for producer in nested:
                if not reads:
                    continue
                for at,release,formal in _unsafe_await_to_discharges(producer):
                    considered+=1
                    for actual,spawn_line in _match_spawn_alias(f,producer,formal):
                        if actual not in reads:
                            continue
                        if not _consumer_reads_after_spawn_without_rebinding(
                            f,exclusions,actual,spawn_line):
                            continue
                        findings.append(Window(
                            source_sha256=digest,
                            caller=f.name,producer=producer.name,
                            spawned_at_line=spawn_line,
                            await_at_line=at,
                            discharge_at_line=release,
                            queue_producer_name=formal,
                            queue_consumer_name=actual,
                        ))
    unique={(w.caller,w.producer,w.spawned_at_line,w.await_at_line,w.discharge_at_line,
             w.queue_producer_name,w.queue_consumer_name):w for w in findings}
    ordered=sorted(unique.values(),key=lambda w:(
        w.caller,w.spawned_at_line,w.await_at_line,w.discharge_at_line,w.producer))
    source_windows=[asdict(x) for x in ordered]
    source_windows.extend(_task_admission_windows(tree,digest))
    source_windows=sorted(source_windows,key=lambda x:(
        x["caller"],x["source_obligation"],
        x.get("spawned_at_line",x.get("guard_at_line",0)),
        x.get("await_at_line",0),
    ))
    return {
        "schema":"obligation_window_ast_v1",
        "source":path,
        "source_sha256":digest,
        "status":"SOURCE_TEST_PLAN_NOT_EXECUTED",
        "candidate_windows":source_windows,
        "candidate_count":len(source_windows),
        "unmatched_pre_discharge_awaits":max(0,considered-len(ordered)),
        "limit":"Only nested async queue producer windows and task collection ID-comparison gates",
    }


def naive_token_baseline(source: str):
    """Deliberately weak sanity comparator, NOT an SOTA baseline."""
    return "await " in source and "put_nowait(None)" in source and ".get()" in source


def choose_by_obligation(windows:list[dict],budget:int):
    """Greedy diversity-only selector; classical set cover, not new theorem."""
    if budget<0:
        raise ValueError("negative budget")
    covered=set()
    pending=list(sorted(windows,key=lambda x:(
        x["caller"],x["source_obligation"],
        x.get("spawned_at_line",x.get("guard_at_line",0)),
        x.get("await_at_line",0))))
    result=[]
    while len(result)<budget and pending:
        scored=[]
        for item in pending:
            key=(item["caller"],item.get("queue_consumer_name",item.get("stored_task_identifier")),
                 item["source_obligation"])
            novelty=int(key not in covered)
            scored.append((novelty,item))
        best=max(s for s,_ in scored)
        if best<=0:
            break
        winner=next(item for novelty,item in scored if novelty==best)
        result.append(winner)
        covered.add((winner["caller"],
                     winner.get("queue_consumer_name",winner.get("stored_task_identifier")),
                     winner["source_obligation"]))
        pending.remove(winner)
    return result


def verify_source_plan(source: str, plan: dict):
    fresh=compile_source(source,path=plan.get("source","<source>"))
    if plan!=fresh:
        raise ValueError("plan not independently reproduced from current source")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--budget",type=int,default=2)
    opts=p.parse_args()
    inp=Path(opts.source)
    result=compile_source(inp.read_text(encoding="utf-8"),path=str(inp))
    result["selected"]=choose_by_obligation(result["candidate_windows"],opts.budget)
    # Replay source selection with all original keys except selected.
    original={k:v for k,v in result.items() if k!="selected"}
    verify_source_plan(inp.read_text(encoding="utf-8"),original)
    dest=Path(opts.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("OBLIGATION_WINDOW_SOURCE",json.dumps({
        "candidate_count":result["candidate_count"],
        "selected":len(result["selected"]),
        "window_sources":[
            (x["caller"], x["source_obligation"],
             x.get("producer",x.get("stored_task_identifier")),
             x.get("await_at_line",x.get("guard_at_line")),
             x.get("discharge_at_line"))
            for x in result["selected"]],
        "source_sha256":result["source_sha256"],
        "status":result["status"],
    },sort_keys=True))
    if not result["selected"]:
        raise SystemExit("NO_SUPPORTED_SOURCE_WINDOWS")


if __name__=="__main__":
    main()
