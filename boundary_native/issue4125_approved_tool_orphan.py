"""Native real SDK reproduction: upstream issue #4125 approved-tool orphaned output.

Upstream issue https://github.com/openai/openai-agents-python/issues/4125
Merged fix https://github.com/openai/openai-agents-python/pull/4126

Reproduces with upstream FakeModel and SimpleListSession, no model API and
no remote effects. Contracts are derived from the author's issue and SDK,
not from raw trace equality (sync vs stream can legitimately differ).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


async def run_case(mode: str, trip: bool):
    from typing import Any
    from agents import (
        Agent, GuardrailFunctionOutput, ModelSettings, OutputGuardrail,
        OutputGuardrailTripwireTriggered, RunContextWrapper, Runner, function_tool,
    )
    from tests.fake_model import FakeModel
    from tests.test_responses import get_function_tool_call
    from tests.utils.simple_session import SimpleListSession

    effect_count = 0

    @function_tool(name_override="secret_tool", needs_approval=True)
    def secret_tool(query: str) -> str:
        nonlocal effect_count
        effect_count += 1
        return "LOCAL_FIXTURE_OUTPUT"

    async def output_guardrail(
        ctx: RunContextWrapper[Any], agent: Agent[Any], output: Any
    ) -> GuardrailFunctionOutput:
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=trip)

    session = SimpleListSession()
    model = FakeModel()
    model.set_next_output([get_function_tool_call("secret_tool", '{"query": "fixture"}')])
    agent = Agent(
        name="local-guardrail-test", model=model, tools=[secret_tool],
        model_settings=ModelSettings(tool_choice="secret_tool"),
        tool_use_behavior="stop_on_first_tool",
        output_guardrails=[OutputGuardrail(guardrail_function=output_guardrail)],
    )

    async def execute(inp):
        if mode == "run":
            return await Runner.run(agent, inp, session=session)
        result = Runner.run_streamed(agent, inp, session=session)
        async for _ in result.stream_events():
            pass
        return result

    first = await execute("Please invoke the local fixture tool")
    state = first.to_state()
    if not first.interruptions:
        raise RuntimeError("expected a human approval interruption; got none")
    state.approve(first.interruptions[0])
    result_status = "ok"
    try:
        await execute(state)
    except OutputGuardrailTripwireTriggered:
        result_status = "blocked"
    items = await session.get_items()
    types = [
        (item.get("type") or item.get("role")) if isinstance(item, dict)
        else type(item).__name__
        for item in items
    ]
    return {
        "mode": mode, "tripwire": trip, "status": result_status,
        "session_types": types, "tool_executions": effect_count,
        "orphaned": ("function_call" in types and
                     types.count("function_call_output") < types.count("function_call")),
        "paired": ("function_call" in types and
                   types.count("function_call_output") == types.count("function_call")),
    }


async def main(args):
    sys.path.insert(0, str(Path(args.upstream_root).resolve()))
    rows = []
    for trip in (False, True):
        for mode in ("run", "streamed"):
            rows.append(await run_case(mode, trip))
    lookup = {(x["mode"], x["tripwire"]): x for x in rows}
    controls = [
        lookup[("run", False)]["status"] == "ok",
        lookup[("streamed", False)]["status"] == "ok",
        lookup[("run", False)]["paired"],
        lookup[("streamed", False)]["paired"],
        all(row["tool_executions"] == 1 for row in rows),
    ]
    if not all(controls):
        verdict = "INVALID_CONTROL"
    elif args.phase == "before":
        verdict = (
            "EXPECTED_HISTORICAL_STREAM_ORPHAN"
            if lookup[("run", True)]["paired"]
            and lookup[("streamed", True)]["orphaned"]
            and lookup[("run", True)]["status"] == "blocked"
            and lookup[("streamed", True)]["status"] == "blocked"
            else "UNEXPECTED_BEFORE_BEHAVIOR"
        )
    else:
        verdict = (
            "EXPECTED_FIXED_APPROVED_EFFECT_PAIR"
            if lookup[("run", True)]["paired"]
            and lookup[("streamed", True)]["paired"]
            and lookup[("run", True)]["status"] == "blocked"
            and lookup[("streamed", True)]["status"] == "blocked"
            else "UNEXPECTED_AFTER_BEHAVIOR"
        )

    result = {
        "source": "openai/openai-agents-python",
        "issue": "https://github.com/openai/openai-agents-python/issues/4125",
        "fix": "https://github.com/openai/openai-agents-python/pull/4126",
        "upstream_commit": args.upstream_commit,
        "state": "NATIVE_HISTORICAL_VERSION",
        "cases": rows,
        "validation": verdict,
        "limits": "one published human-approved local fake tool, no remote effects or novel method",
    }
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("ORPHAN_REPRO", json.dumps(result, sort_keys=True), flush=True)
    if verdict.startswith(("INVALID_", "UNEXPECTED_")):
        raise SystemExit(2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--upstream-root", required=True)
    p.add_argument("--upstream-commit", required=True)
    p.add_argument("--phase", choices=("before", "after"), required=True)
    p.add_argument("--output", required=True)
    asyncio.run(main(p.parse_args()))
