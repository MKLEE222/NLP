"""Execute source-derived obligation-window pair through an audited native SDK adapter.

The source analyzer does not claim to know public SDK contracts, dispatch an
LLM, or insert arbitrary patch code. This driver explicitly accepts one
supported, public native AutoGen adapter and validates the recompiled plan
before running real SDK code with a deterministic in-process tool.

The 12 sealed-confirmation issues are not loaded.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from obligation_window_kernel import choose_by_obligation, verify_source_plan


def verify_manifest(root: Path, path: str, doc: dict):
    source=(root / path).resolve()
    if not source.is_relative_to(root.resolve()):
        raise ValueError("source path escapes pinned checkout")
    if not source.is_file():
        raise ValueError("source is not a file")
    plan={k:v for k,v in doc.items() if k!="selected"}
    plan["source"]=doc.get("source")
    verify_source_plan(source.read_text(encoding="utf-8"),plan)
    expected=choose_by_obligation(plan["candidate_windows"],len(doc["selected"]))
    if doc["selected"]!=expected:
        raise ValueError("generated window selection has been altered")
    if not doc["selected"]:
        raise ValueError("no obligation window to execute")
    return doc["selected"]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--plan",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--timeout",type=float,default=1.5)
    args=p.parse_args()
    root=Path(args.root).resolve()
    rel="python/packages/autogen-agentchat/src/autogen_agentchat/agents/_assistant_agent.py"
    doc=json.loads(Path(args.plan).read_text())
    if Path(doc.get("source","")).as_posix() not in (rel,"upstream/"+rel):
        raise ValueError("the manually trusted adapter only handles pinned AutoGen source")
    chosen=verify_manifest(root,rel,doc)
    if not (len(chosen)==1
            and chosen[0]["source_obligation"]=="stream_terminal_delivery"
            and chosen[0]["fault"]=="cancel_at_producer_await"
            and chosen[0]["queue_producer_name"]=="stream_queue"
            and chosen[0]["queue_consumer_name"]=="stream"):
        raise ValueError("unsupported observable contract; abstain")
    from execute_autogen_cancel_plan import native_test
    if not (0.1<=args.timeout<=5):
        raise ValueError("timeout outside permitted bound")
    result=asyncio.run(native_test(args.timeout))
    receipt={
        "status":"SOURCE_TO_NATIVE_EXECUTED",
        "source_digest":chosen[0]["source_sha256"],
        "activation_line":chosen[0]["spawned_at_line"],
        "fault_cut_line":chosen[0]["await_at_line"],
        "discharge_line":chosen[0]["discharge_at_line"],
        "obligation":chosen[0]["source_obligation"],
        "observation":result,
        "disclaimer":"known public AutoGen development bug; manually trusted native SDK adapter",
    }
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print("WINDOW_NATIVE",json.dumps(receipt,sort_keys=True))
    if result["status"]=="UNKNOWN":
        raise SystemExit("unknown runtime observation")


if __name__=="__main__":
    main()
