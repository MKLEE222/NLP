"""Bounded offline native execution of one source-planned AutoGen cancellation test.

This is a DEVELOPMENT issue opened after the 16-issue split was committed.
No paid API, no network effects, and no unbounded hang: tool and consumer both
use explicit controls and timeouts. A cancellation exception is a legal terminal
signal; a task remaining blocked beyond the budget is a FAIL only when the
normal success control and tool-start synchronization are valid.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from source_guided_synth import verify_pinned_source


async def _run_variant(cancel_inflight: bool, timeout: float):
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import TextMessage
    from autogen_core import CancellationToken, FunctionCall
    from autogen_core.models import CreateResult, ModelFamily, RequestUsage
    from autogen_ext.models.replay import ReplayChatCompletionClient

    entered=asyncio.Event()
    release=asyncio.Event()

    async def fixture_tool() -> str:
        """Private local instrument with no external I/O."""
        entered.set()
        await release.wait()
        return "local-success"

    response=CreateResult(
        content=[FunctionCall(id="local-call-1",name="fixture_tool",arguments="{}")],
        finish_reason="function_calls",
        usage=RequestUsage(prompt_tokens=1,completion_tokens=1),
        cached=False,
    )
    client=ReplayChatCompletionClient(
        [response],
        model_info={"function_calling":True,"vision":False,"json_output":False,
                    "family":ModelFamily.GPT_4O,"structured_output":False},
    )
    agent=AssistantAgent("native_cancel_fixture",model_client=client,
                         tools=[fixture_tool],reflect_on_tool_use=False)
    token=CancellationToken()
    events=[]
    async def consume():
        async for item in agent.on_messages_stream(
            [TextMessage(content="run exactly the local tool",source="user")],token
        ):
            events.append(type(item).__name__)
    task=asyncio.create_task(consume())
    terminal="unknown"
    thrown=None
    started=False
    try:
        await asyncio.wait_for(entered.wait(),timeout=timeout)
        started=True
        if cancel_inflight:
            token.cancel()
        else:
            release.set()
        try:
            await asyncio.wait_for(task,timeout=timeout)
            terminal="returned"
        except asyncio.TimeoutError:
            terminal="timeout"
        except asyncio.CancelledError:
            terminal="cancelled_exception"
        except Exception as exc:
            thrown=type(exc).__name__
            terminal="exception"
    except asyncio.TimeoutError:
        terminal="fixture_tool_not_reached"
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task,return_exceptions=True)
        await client.close()
    return {
        "variant":"cancel_inflight" if cancel_inflight else "normal_complete",
        "tool_started":started,
        "terminal":terminal,
        "exception_type":thrown,
        "events":events,
    }


async def native_test(timeout:float):
    control=await _run_variant(False,timeout)
    fault=await _run_variant(True,timeout)
    if not (control["tool_started"] and control["terminal"]=="returned"
            and "ToolCallExecutionEvent" in control["events"]):
        verdict="UNKNOWN"
        reason="no_valid_success_control"
    elif not fault["tool_started"]:
        verdict="UNKNOWN"
        reason="cancelled_before_tool_start"
    elif fault["terminal"]=="timeout":
        verdict="FAIL"
        reason="cancellation_blocks_stream_until_timeout"
    elif fault["terminal"] in ("returned","cancelled_exception"):
        verdict="PASS"
        reason="cancellation_has_bounded_terminal_signal"
    else:
        verdict="UNKNOWN"
        reason="unexplained_exception_or_unsupported_execution"
    return {"status":verdict,"reason":reason,"control":control,"fault":fault,
            "timeout_seconds_per_phase":timeout}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--upstream-root",required=True)
    p.add_argument("--plan",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--timeout",type=float,default=1.0)
    args=p.parse_args()
    if not (0.1 <= args.timeout <= 5.0):
        raise ValueError("test timeout must be bounded 0.1-5s")
    root=Path(args.upstream_root)
    plan=json.loads(Path(args.plan).read_text())
    verify_pinned_source(root,plan)
    if plan["profile"]!="autogen-tool-cancellation" or len(plan["selected"])!=1:
        raise ValueError("unsupported profile or scenario count")
    selected=plan["selected"][0]
    if (selected["adapter"]!="autogen-cancellation-stream"
        or selected["injection"]["variants"]!=["normal_complete","cancel_inflight"]):
        raise ValueError("unsupported source-derived variant")
    observation=asyncio.run(native_test(args.timeout))
    record={
        "profile":plan["profile"],
        "pinned_source_sha256":plan["source_sha256"],
        "chosen_source_line":selected["source_line"],
        "producer_function":selected["producer"],
        "claim_limit":"frozen development issue on one upstream commit, not a blind new discovery",
        "native_result":observation,
    }
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print("CANCEL_STREAM_NATIVE",json.dumps(record,sort_keys=True),flush=True)
    if observation["status"]=="UNKNOWN":
        raise SystemExit("unknown native observation is not a PASS")


if __name__=="__main__":
    main()
