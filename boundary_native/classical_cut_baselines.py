"""Fairly described *local adaptations* of classical fault-cut selection ideas.

B0: deterministic uniform fault-site choice
B1: source-local await/terminal or pending-ID comparison
B2: source-level lineage from spawned producer resource to awaited consumer
B3: healthy-trace-guided fault-site order, ONLY when an actual line trace exists

B2 is NOT SIGMOD 2015 Molly/LDFI. B3 is NOT OSDI 2022 Sieve.
Neither reproduces full causal-SAT reasoning nor Kubernetes controller
instrumentation, safety/liveness oracles and fault-replay engine.

All use the SAME source text and independent raw fault-cut enumeration.
No bug/fix labels, issue numbers or expected native verdict enter methods.
"""
from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from typing import Any


SEED="OD-DEV-20261011-V1"


def _calls(fn):
    out=set()
    for x in ast.walk(fn):
        if isinstance(x,ast.Call):
            name=x.func.id if isinstance(x.func,ast.Name) else (
                x.func.attr if isinstance(x.func,ast.Attribute) else None)
            if name:
                out.add(name)
    return out


def _candidate_awaits_in_function(fn):
    return sorted({x.lineno for x in ast.walk(fn)
                   if isinstance(x,ast.Await)})


def raw_sites(source: str) -> list[dict]:
    """All structurally available await and pending-ID comparison *cuts*.

    No stream obligation inference and no source bug label. These are the
    shared syntax candidate universe for B0/B3 and denominator auditing.
    """
    tree=ast.parse(source)
    result={}
    for fn in ast.walk(tree):
        if not isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)):
            continue
        name=fn.name
        for node in ast.walk(fn):
            if isinstance(node,ast.Await):
                key=("await",name,node.lineno)
                result[key]={"kind":"await","caller":name,"line":node.lineno}
            if isinstance(node,ast.Compare):
                if (len(node.ops)==1 and isinstance(node.ops[0],(ast.Eq,ast.Is))
                    and len(node.comparators)==1
                    and isinstance(node.comparators[0],ast.Attribute)
                    and node.comparators[0].attr=="id"):
                    key=("id_guard",name,node.lineno)
                    result[key]={"kind":"id_guard","caller":name,"line":node.lineno}
    return sorted(result.values(),key=lambda x:(x["line"],x["caller"],x["kind"]))


def _direct_sentinel(stmt):
    if not isinstance(stmt,ast.Expr) or not isinstance(stmt.value,ast.Call):
        return None
    c=stmt.value
    if not (isinstance(c.func,ast.Attribute) and c.func.attr in ("put_nowait","put")
            and isinstance(c.func.value,ast.Name) and len(c.args)==1
            and isinstance(c.args[0],ast.Constant) and c.args[0].value is None):
        return None
    return c.func.value.id


def _nested_async(tree):
    for outer in ast.walk(tree):
        if not isinstance(outer,ast.AsyncFunctionDef):
            continue
        inner=[x for x in ast.walk(outer)
               if isinstance(x,ast.AsyncFunctionDef) and x is not outer]
        for producer in inner:
            yield outer,producer


def _task_id_gate(tree):
    """A simple scheduling-context ID-guard source baseline."""
    found=[]
    for fn in ast.walk(tree):
        if not isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)):
            continue
        if "schedule_task" not in _calls(fn):
            continue
        for cmp in ast.walk(fn):
            if not (isinstance(cmp,ast.Compare) and len(cmp.ops)==1
                    and isinstance(cmp.ops[0],(ast.Eq,ast.Is))
                    and len(cmp.comparators)==1):
                continue
            rhs=cmp.comparators[0]
            if not (isinstance(rhs,ast.Attribute) and rhs.attr=="id"):
                continue
            found.append({"caller":fn.name,"line":cmp.lineno,
                          "kind":"inflight_logical_task_not_reissued",
                          "method_detail":"local scheduling id equality"})
    return found


def source_local_windows(source: str) -> list[dict]:
    """B1: nearby await and terminal, without queue identity or cleanup."""
    tree=ast.parse(source)
    result=_task_id_gate(tree)
    for outer,inner in _nested_async(tree):
        # Notice that this intentionally scans ANY reachable AST node:
        # an exception-protected finally will be flagged as well.
        awaits=_candidate_awaits_in_function(inner)
        sinks=[(n.lineno,_direct_sentinel(n)) for n in ast.walk(inner)
               if _direct_sentinel(n) is not None]
        for at in awaits:
            if any(end>at for end,q in sinks):
                result.append({
                    "caller":outer.name,"line":at,"kind":"stream_terminal_delivery",
                    "method_detail":"nested await before local none sentinel",
                })
    return _unique(result)


def _spawn_alias(outer, producer, formal):
    params=[x.arg for x in producer.args.posonlyargs+producer.args.args]
    if formal not in params:
        return []
    idx=params.index(formal)
    result=[]
    for node in ast.walk(outer):
        if not isinstance(node,ast.Call):
            continue
        if not (isinstance(node.func,ast.Attribute)
                and node.func.attr=="create_task"):
            continue
        for sub in node.args:
            if not (isinstance(sub,ast.Call) and isinstance(sub.func,ast.Name)
                    and sub.func.id==producer.name):
                continue
            actual=(sub.args[idx] if idx<len(sub.args) else
                    next((kw.value for kw in sub.keywords if kw.arg==formal),None))
            if isinstance(actual,ast.Name):
                result.append(actual.id)
    return result


def source_lineage_windows(source: str) -> list[dict]:
    """B2: source lineage relation, but no finally/early-return proof.

    This is a transparent *approximation* inspired by lineage-directed
    candidates and unrelated to Molly's original constraint solver.
    """
    tree=ast.parse(source)
    result=_task_id_gate(tree)
    for outer,producer in _nested_async(tree):
        inner_ids={id(x) for x in ast.walk(producer)}
        awaited_queues={
            v.func.value.id for a in ast.walk(outer)
            if id(a) not in inner_ids and isinstance(a,ast.Await)
            and isinstance(a.value,ast.Call)
            and isinstance(a.value.func,ast.Attribute)
            and a.value.func.attr=="get"
            and isinstance(a.value.func.value,ast.Name)
            for v in [a.value]
        }
        if not awaited_queues:
            continue
        awaits=_candidate_awaits_in_function(producer)
        sinks=[(node.lineno,formal) for node in ast.walk(producer)
               if (formal:=_direct_sentinel(node)) is not None]
        for at in awaits:
            for ending,formal in sinks:
                if ending<=at:
                    continue
                if not set(_spawn_alias(outer,producer,formal)) & awaited_queues:
                    continue
                result.append({
                    "caller":outer.name,"line":at,"kind":"stream_terminal_delivery",
                    "method_detail":"local source producer-consumer lineage",
                })
    return _unique(result)


def _unique(candidates):
    seen={}
    for x in candidates:
        key=(x["caller"],x["line"],x["kind"])
        seen[key]=x
    return sorted(seen.values(),key=lambda x:(x["line"],x["caller"],x["kind"]))


def uniform_sites(source: str, budget: int, *, seed=SEED) -> list[dict]:
    if budget<0:
        raise ValueError("negative budget")
    candidates=raw_sites(source)
    # Stable across OS/Python; avoids process-randomized hash().
    keyed=lambda x: hashlib.sha256(
        (seed+"|"+x["kind"]+"|"+x["caller"]+"|"+str(x["line"])).encode()).hexdigest()
    return sorted(candidates,key=lambda x:(keyed(x),x["line"]))[:budget]


def trace_guided_sites(
    source: str, observed_lines: set[int] | None,
    budget: int
) -> dict:
    """B3 *only* when a real healthy-run source-line trace is supplied."""
    if budget<0:
        raise ValueError("negative budget")
    if observed_lines is None:
        return {"status":"UNSUPPORTED_NO_REAL_HEALTHY_TRACE","selected":[]}
    if any(type(x) is not int or x<1 for x in observed_lines):
        raise ValueError("invalid source line trace")
    candidates=[x for x in raw_sites(source) if x["line"] in observed_lines]
    # Simplified later-in-observed-trace priority; source line is a proxy,
    # not the actual Sieve algorithm's controller event order.
    candidates.sort(key=lambda x:(-x["line"],x["caller"],x["kind"]))
    return {"status":"RANKED_FROM_REAL_HEALTHY_TRACE",
            "selected":candidates[:budget],
            "eligible_observed":len(candidates)}


def method_summary(source: str, budget: int, *, observed_lines=None) -> dict:
    local=source_local_windows(source)
    lineage=source_lineage_windows(source)
    trace=trace_guided_sites(source,observed_lines,budget)
    return {
        "raw_cut_denominator":len(raw_sites(source)),
        "B0_seeded_uniform":uniform_sites(source,budget),
        "B1_source_local":{"candidate_count":len(local),"selected":local[:budget]},
        "B2_lineage_inspired":{"candidate_count":len(lineage),"selected":lineage[:budget]},
        "B3_trace_inspired":trace,
        "warning":"B2/B3 ARE NOT OFFICIAL LDFI OR SIEVE; only same-source developer adapters",
    }
