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
                formal=_is_none_sentinel(later)
                if formal is not None and formal not in guarded:
                    for await_site in awaits:
                        windows.append((await_site.lineno,later.lineno,formal))
    walk_block(producer.body,set())
    return windows


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
    return {
        "schema":"obligation_window_ast_v0",
        "source":path,
        "source_sha256":digest,
        "status":"SOURCE_TEST_PLAN_NOT_EXECUTED",
        "candidate_windows":[asdict(x) for x in ordered],
        "candidate_count":len(ordered),
        "unmatched_pre_discharge_awaits":max(0,considered-len(ordered)),
        "limit":"Only nested async producers and explicit positional/keyword create_task(queue) aliases",
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
        x["caller"],x["spawned_at_line"],x["await_at_line"])))
    result=[]
    while len(result)<budget and pending:
        scored=[]
        for item in pending:
            key=(item["caller"],item["queue_consumer_name"],item["source_obligation"])
            novelty=int(key not in covered)
            scored.append((novelty,item))
        best=max(s for s,_ in scored)
        if best<=0:
            break
        winner=next(item for novelty,item in scored if novelty==best)
        result.append(winner)
        covered.add((winner["caller"],winner["queue_consumer_name"],winner["source_obligation"]))
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
        "window_sources":[(x["caller"],x["producer"],x["await_at_line"],
                           x["discharge_at_line"]) for x in result["selected"]],
        "source_sha256":result["source_sha256"],
        "status":result["status"],
    },sort_keys=True))
    if not result["selected"]:
        raise SystemExit("NO_SUPPORTED_SOURCE_WINDOWS")


if __name__=="__main__":
    main()
