"""Native public SDK regression probe for openai-agents-python issue #3280.

Runs the source author's repro against fixed historical commits with FakeModel.
Never calls a hosted LLM or writes to remote services. The script captures
effects of guardrail exceptions on stream terminal state; it is NOT an
original testing method or a representative sample of production defects.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path


async def run_case(exc: bool):
    from agents import (
        Agent,
        GuardrailFunctionOutput,
        OutputGuardrail,
        RunContextWrapper,
        Runner,
    )
    from tests.fake_model import FakeModel
    from tests.test_responses import get_text_message

    def guardrail_function(ctx: RunContextWrapper, agent: Agent, out: object):
        if exc:
            raise RuntimeError("fixture_guardrail_failure")
        return GuardrailFunctionOutput(output_info="ok", tripwire_triggered=False)

    model = FakeModel(initial_output=[get_text_message("fixture_answer")])
    agent = Agent(
        name="local-no-network-agent",
        model=model,
        output_guardrails=[OutputGuardrail(guardrail_function=guardrail_function)],
    )
    result = Runner.run_streamed(agent, input="fixture_user")
    error_type = None
    error_fingerprint = None
    try:
        await result.run_loop_task
    except Exception as e:
        error_type = type(e).__name__
        error_fingerprint = hashlib.sha256(str(e).encode("utf-8")).hexdigest()[:16]
    return {
        "exception_injected": exc,
        "caught_exception": error_type,
        "exception_message_sha16": error_fingerprint,
        "is_complete": result.is_complete,
        "final_output_accepted": result.final_output is not None,
        "final_output_value_sha16": (
            hashlib.sha256(str(result.final_output).encode("utf-8")).hexdigest()[:16]
            if result.final_output is not None else None
        ),
    }


async def main(args):
    sys.path.insert(0, str(Path(args.upstream_root).resolve()))
    import agents
    baseline = await run_case(False)
    injected = await run_case(True)
    report = {
        "source": "openai/openai-agents-python",
        "upstream_commit": args.upstream_commit,
        "historical_issue": "https://github.com/openai/openai-agents-python/issues/3280",
        "merge_fix": "https://github.com/openai/openai-agents-python/pull/3281",
        "execution": "native-installed-upstream-deterministic-FakeModel",
        "contract": "raised output guardrail exception must not be treated as success",
        "control": baseline,
        "fault": injected,
        "status": "EXECUTED_NATIVE_HISTORICAL_VERSION",
        "not_a_claim": "single published bug reproduction; no new causal testing algorithm",
    }
    if not baseline["is_complete"] or not baseline["final_output_accepted"]:
        report["validation"] = "INVALID_CONTROL"
    elif args.phase == "before":
        report["validation"] = (
            "EXPECTED_HISTORICAL_FAIL_OPEN"
            if injected["caught_exception"] is None and injected["final_output_accepted"]
            else "UNEXPECTED_BEFORE_BEHAVIOR"
        )
    else:
        report["validation"] = (
            "EXPECTED_FIXED_EXCEPTION_PROPAGATION"
            if injected["caught_exception"] == "RuntimeError" and not injected["final_output_accepted"]
            else "UNEXPECTED_AFTER_BEHAVIOR"
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("NATIVE_REPRO", json.dumps(report, sort_keys=True), flush=True)
    if report["validation"].startswith(("INVALID_", "UNEXPECTED_")):
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--output", required=True)
    asyncio.run(main(parser.parse_args()))
