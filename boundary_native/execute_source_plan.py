"""Execute an automatically planned, bounded native SDK failure-boundary probe.

Profiles are explicit, separately reviewable adapters. The plan chooses legal
fault variants and a source location; adapters provide SDK-specific objects.
No LLM endpoint, remote tool, or production side effect is reachable.
Independent runtime verdicts do not read source handler labels.
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import sys
import weakref
from pathlib import Path

from source_guided_synth import verify_pinned_source


async def _agent_case(fault: str):
    from agents import Agent, GuardrailFunctionOutput, OutputGuardrail, Runner
    from tests.fake_model import FakeModel
    from tests.test_responses import get_text_message

    def guardrail(*args):
        if fault=="raise_RuntimeError":
            raise RuntimeError("source_generated_exception")
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    agent=Agent(name="source-generated-local-fixture",
                model=FakeModel(initial_output=[get_text_message("fixture_answer")]),
                output_guardrails=[OutputGuardrail(guardrail_function=guardrail)])
    result=Runner.run_streamed(agent, input="fixture_input")
    exc_type=None
    try:
        await result.run_loop_task
    except Exception as e:
        exc_type=type(e).__name__
    return {
        "variant":fault,
        "caught_exception":exc_type,
        "final_output_accepted":result.final_output is not None,
        "is_complete":bool(result.is_complete),
    }


async def _run_agents(candidate):
    variants=candidate["injection"]["variants"]
    if variants!=["pass","raise_RuntimeError"]:
        raise ValueError("plan requires unsupported guardrail variants")
    control=await _agent_case(variants[0])
    fault=await _agent_case(variants[1])
    if control["caught_exception"] is not None or not control["final_output_accepted"]:
        return {"status":"UNKNOWN","reason":"invalid_pass_control",
                "control":control,"fault":fault}
    violation=fault["final_output_accepted"] or fault["caught_exception"]!="RuntimeError"
    return {
        "status":"FAIL" if violation else "PASS",
        "reason":("exception_swallowed_or_output_accepted" if violation
                  else "exception_propagated_and_final_rejected"),
        "control":control,"fault":fault,
    }


def _langgraph_case(same):
    from langgraph._internal._constants import CONF, CONFIG_KEY_SCRATCHPAD
    from langgraph.pregel._runner import _call

    class Task:
        pass

    class Map(dict):
        pass

    class Scratch:
        def __init__(self):
            self.i=0
        def call_counter(self):
            self.i+=1
            return self.i

    parent=Task()
    parent.id="parent-logical"
    parent.config={CONF:{CONFIG_KEY_SCRATCHPAD:Scratch()}}
    existing=Task()
    existing.id="logical-child-A"
    existing.writes=[]
    next_child=Task()
    next_child.id="logical-child-A" if same else "independent-child-B"
    next_child.writes=[]
    pending=Map()
    previous=concurrent.futures.Future()
    pending[previous]=existing
    admitted=[]

    def schedule_task(parent_task,counter,call):
        return next_child

    def submit(*args,**kwargs):
        admitted.append(next_child.id)
        return concurrent.futures.Future()

    chain=_call(weakref.ref(parent),lambda _:None,None,
                futures=weakref.ref(pending),
                schedule_task=schedule_task,submit=weakref.ref(submit))
    return {
        "variant":"same_logical_id" if same else "different_logical_id",
        "original_future_pending":not previous.done(),
        "admissions":len(admitted),
        "returned_future_pending":not chain.done(),
    }


def _run_langgraph(candidate):
    variants=candidate["injection"]["variants"]
    if variants!=["same_logical_id","different_logical_id"]:
        raise ValueError("plan requires unsupported admission variants")
    same=_langgraph_case(True)
    different=_langgraph_case(False)
    if not (same["original_future_pending"] and different["original_future_pending"]
            and different["admissions"]==1):
        return {"status":"UNKNOWN","reason":"invalid_different_id_control",
                "same":same,"different":different}
    violation=same["admissions"]>0
    return {
        "status":"FAIL" if violation else "PASS",
        "reason":"duplicate_inflight_child_admitted" if violation else "pending_child_reused",
        "same":same,"different":different,
    }


def run(root:Path,plan:dict):
    verify_pinned_source(root,plan)
    selected=plan["selected"]
    if len(selected)!=1:
        raise ValueError("one source-justified scenario required")
    case=selected[0]
    if not case.get("eligible") or case["adapter"]!=plan["adapter"]:
        raise ValueError("unsupported/invalid case")
    if plan["profile"]=="agents-guardrail":
        sys.path.insert(0,str(root.resolve()))
        outcome=asyncio.run(_run_agents(case))
    else:
        sys.path.insert(0,str(root.resolve()/"libs"/"langgraph"))
        outcome=_run_langgraph(case)
    return {
        "source_profile":plan["profile"],
        "source_file":plan["source_file"],
        "source_sha256":plan["source_sha256"],
        "generated_at_function":case["function"],
        "generated_at_line":case["source_line"],
        "source_hint":case.get("handler_exit",case.get("comparison")),
        "plan_status":plan["status"],
        "runtime_result":outcome,
        "evidence_limit":"one generated known-profile native scenario, not unrestricted source-to-test synthesis",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--upstream-root",required=True)
    p.add_argument("--plan",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    outcome=run(Path(args.upstream_root),json.loads(Path(args.plan).read_text()))
    dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(outcome,indent=2,sort_keys=True)+"\n")
    print("SOURCE_PLAN_EXEC",json.dumps(outcome,sort_keys=True),flush=True)
    if outcome["runtime_result"]["status"]=="UNKNOWN":
        raise SystemExit("INVALID_NATIVE_CONTROL_OR_UNKNOWN_CONTRACT")


if __name__=="__main__":
    main()
