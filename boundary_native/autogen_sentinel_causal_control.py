"""Guarded test-only causal intervention on one preregistered AutoGen DEV source.

Move the tool stream's end-of-queue signal into a finally block. This is a
diagnostic intervention, NOT an upstream fix proposal: it does not implement
proper sibling task cleanup, cancel propagation or return status semantics.
Only tests whether a guaranteed queue terminator removes the hang.

The original pinned commit, source digest and AST line are required; any
unexpected file or missing topology fails closed without patching.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from source_guided_synth import synthesize

PINNED_RELATIVE="python/packages/autogen-agentchat/src/autogen_agentchat/agents/_assistant_agent.py"
ORIGINAL_SHA256="c0e578fc34be40f46f8e36a91b53fb27cf5ff454f4efb3f4ea5dc1e32d0048c1"


def guarded_patch(root:Path):
    source=root / PINNED_RELATIVE
    orig=source.read_bytes()
    digest=hashlib.sha256(orig).hexdigest()
    if digest!=ORIGINAL_SHA256:
        raise ValueError(f"wrong upstream source digest {digest}")
    original_plan=synthesize(root,"autogen-tool-cancellation",budget=1)
    if len(original_plan["selected"])!=1:
        raise ValueError("no unique pre-patch source-selected cancellation")
    case=original_plan["selected"][0]
    if case["producer"]!="_execute_tool_calls":
        raise ValueError("wrong source producer")
    if case.get("queue_flow")!={"producer_queue":"stream_queue","consumer_queue":"stream"}:
        raise ValueError("producer-to-consumer queue was not source-verified")
    tree=ast.parse(orig.decode("utf-8"),filename=PINNED_RELATIVE)
    producers=[x for x in ast.walk(tree)
               if isinstance(x,ast.AsyncFunctionDef) and x.name=="_execute_tool_calls"]
    if len(producers)!=1:
        raise ValueError("expected exactly one SDK producer")
    body=producers[0].body
    matched=[]
    for i,stmt in enumerate(body[:-1]):
        nxt=body[i+1]
        if (isinstance(stmt,(ast.Assign,ast.Expr))
            and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)
                    and x.func.attr=="gather" for x in ast.walk(stmt))
            and isinstance(nxt,ast.Expr)
            and isinstance(nxt.value,ast.Call)
            and isinstance(nxt.value.func,ast.Attribute)
            and nxt.value.func.attr=="put_nowait"
            and len(nxt.value.args)==1
            and isinstance(nxt.value.args[0],ast.Constant)
            and nxt.value.args[0].value is None):
            matched.append((stmt,nxt))
    if len(matched)!=1:
        raise ValueError("expected exactly one gather-to-sentinel direct adjacency")
    gather,sentinel=matched[0]
    if gather.lineno!=case["source_line"] or sentinel.lineno!=case["terminal_line"]:
        raise ValueError("source plan and AST do not agree")
    lines=orig.decode("utf-8").splitlines(keepends=True)
    assignment=lines[gather.lineno-1:gather.end_lineno]
    sentinel_lines=lines[sentinel.lineno-1:sentinel.end_lineno]
    indent=" "*(gather.col_offset)
    # Original source's intervening comment is discarded from the *test-only*
    # local checkout. The semantic transformation is explicit and reviewable.
    translated=[
        indent+"try:\n",
        *["    "+line for line in assignment],
        indent+"finally:\n",
        *["    "+line for line in sentinel_lines],
    ]
    newlines=lines[:gather.lineno-1]+translated+lines[sentinel.end_lineno:]
    candidate="".join(newlines)
    ast.parse(candidate,filename=PINNED_RELATIVE)
    source.write_text(candidate)
    post_plan=synthesize(root,"autogen-tool-cancellation",budget=1)
    if post_plan["selected"]:
        raise ValueError("test-only repaired source still selected as missing sentinel")
    return {
        "upstream_original_source_sha256":digest,
        "patched_test_only_sha256":hashlib.sha256(candidate.encode()).hexdigest(),
        "original_selected_line":gather.lineno,
        "original_sentinel_line":sentinel.lineno,
        "test_only_transform":"guaranteed_queue_terminator_in_finally",
        "post_plan_eligible":len(post_plan["selected"]),
        "claim_limit":"in-container patch, NOT merged upstream nor a complete concurrency fix",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--upstream-root",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    data=guarded_patch(Path(args.upstream_root))
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
    print("CAUSAL_CONTROL",json.dumps(data,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
