"""Boundary-obligation prototype on source-native Agent SDK observations.

This is a reusable EVALUATION COMPONENT, not a new algorithm or a verified
synthesis method. It ignores the source fixtures' expected verdict labels.
Each obligation must be explicitly grounded in upstream documented semantics.
Only public, deterministic native traces are accepted by the current driver.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"


@dataclass(frozen=True)
class Result:
    channel: str
    status: str
    reasons: tuple[str, ...]

    def to_dict(self):
        return {"channel": self.channel, "status": self.status,
                "reasons": list(self.reasons)}


def _require_bool(row: dict, key: str) -> bool:
    x = row.get(key)
    if type(x) is not bool:
        raise ValueError(f"missing/not-boolean observation: {key}")
    return x


def check_guardrail_exception(o: dict[str, Any]) -> Result:
    """If a known output guardrail throws, final acceptance is invalid.

    NOTE: this is a contract of upstream SDK issue #3280 and is not a claim
    that every SDK exception has the same contractual interpretation.
    """
    try:
        control, fault = o["control"], o["fault"]
        if _require_bool(control, "exception_injected") or not _require_bool(
            control, "final_output_accepted"
        ):
            return Result("guardrail_exception", UNKNOWN, ("control_invalid",))
        if not _require_bool(fault, "exception_injected"):
            return Result("guardrail_exception", UNKNOWN, ("fault_not_injected",))
        accepted = _require_bool(fault, "final_output_accepted")
        caught = fault.get("caught_exception")
        if caught is not None and not isinstance(caught, str):
            raise ValueError("caught_exception must be str or null")
        reasons = []
        if accepted:
            reasons.append("rejected_guardrail_output_accepted")
        if caught is None:
            reasons.append("injected_exception_not_propagated")
        return Result("guardrail_exception", FAIL if reasons else PASS,
                      tuple(reasons) or ("exception_propagated_no_final_output",))
    except (TypeError, KeyError, ValueError) as e:
        return Result("guardrail_exception", UNKNOWN, ("invalid_observation:" + str(e),))


def check_approved_result_pair(o: dict[str, Any]) -> Result:
    """Test durable call/result pairing under *this issue's* approval contract.

    Actual tool effect count is local fake-tool count; a remote commit would
    require signed/idempotent receipt or UNKNOWN, not inferred exactly-once.
    """
    try:
        cases = o["cases"]
        if len(cases) != 4:
            return Result("approved_result", UNKNOWN, ("missing_mode_guardrail_controls",))
        seen = {}
        for case in cases:
            k = (case["mode"], _require_bool(case, "tripwire"))
            if k in seen or case["mode"] not in ("run", "streamed"):
                return Result("approved_result", UNKNOWN, ("duplicate_or_invalid_case",))
            seen[k] = case
        needed = {("run", False), ("run", True),
                  ("streamed", False), ("streamed", True)}
        if set(seen) != needed:
            return Result("approved_result", UNKNOWN, ("uncovered_conditions",))
        issues = []
        for (mode, trip), case in seen.items():
            if case.get("status") != ("blocked" if trip else "ok"):
                issues.append(f"{mode}:{trip}:wrong_final_status")
            # Count only raw session items, NOT fixture-computed 'orphaned'.
            types = case["session_types"]
            calls = types.count("function_call")
            outputs = types.count("function_call_output")
            effects = case.get("tool_executions")
            if calls != 1 or effects != 1:
                return Result("approved_result", UNKNOWN,
                              ("unexpected_tool_invocation_or_effect_count",))
            if outputs != calls:
                issues.append(f"{mode}:{trip}:committed_call_without_result")
        return Result("approved_result", FAIL if issues else PASS,
                      tuple(issues) or ("approved_effect_output_paired",))
    except (TypeError, KeyError, ValueError) as e:
        return Result("approved_result", UNKNOWN, ("invalid_observation:" + str(e),))


def check_inflight_dedup(o: dict[str, Any]) -> Result:
    """At most one new child admission per logical ID already in flight."""
    try:
        same = o["same_id"]
        different = o["different_id_negative_control"]
        if not (_require_bool(same, "logical_id_same") and
                not _require_bool(different, "logical_id_same")):
            return Result("inflight_dedup", UNKNOWN, ("id_contrast_invalid",))
        if not (_require_bool(same, "original_pending_future") and
                _require_bool(different, "original_pending_future")):
            return Result("inflight_dedup", UNKNOWN, ("no_live_existing_child",))
        if different["new_child_admissions"] != 1:
            return Result("inflight_dedup", UNKNOWN, ("independent_child_control_invalid",))
        count = same["new_child_admissions"]
        if type(count) is not int or count < 0:
            return Result("inflight_dedup", UNKNOWN, ("invalid_admission_count",))
        return Result("inflight_dedup", FAIL if count else PASS,
                      ("duplicate_same_logical_id",) if count else ("same_id_reused",))
    except (TypeError, KeyError, ValueError) as e:
        return Result("inflight_dedup", UNKNOWN, ("invalid_observation:" + str(e),))


CHECKS = {
    "guardrail_exception": check_guardrail_exception,
    "approved_result": check_approved_result_pair,
    "inflight_dedup": check_inflight_dedup,
}


def judge(row: dict) -> Result:
    channel = row.get("channel")
    obs = row.get("observation")
    if channel not in CHECKS or not isinstance(obs, dict):
        return Result(str(channel), UNKNOWN, ("unrecognized_channel_or_observation",))
    return CHECKS[channel](obs)


def evaluate_evidence(path: Path):
    src = json.loads(path.read_text())
    results = []
    for row in src["rows"]:
        verdict = judge(row)
        results.append({
            "run": row["run"], "job": row["job"],
            "source": row["observation"].get("source"),
            "sha": row["observation"].get(
                "upstream_commit", row["observation"].get("source_sha")),
            **verdict.to_dict(),
        })
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    a=p.parse_args()
    results=evaluate_evidence(Path(a.input))
    counts={s: sum(x["status"]==s for x in results) for s in (PASS,FAIL,UNKNOWN)}
    report={
        "object":"native-source-boundary-obligation-validation",
        "not_a_novelty_claim":"Three hand-grounded obligation types; no automatic test synthesis yet.",
        "real_native_observations":len(results),
        "statuses":counts,
        "results":results,
    }
    target=Path(a.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("OBLIGATION_RESULTS",json.dumps(report,sort_keys=True),flush=True)
    if len(results)!=6 or counts[PASS]!=3 or counts[FAIL]!=3 or counts[UNKNOWN]:
        raise SystemExit("Observed source/fix paired reproduction no longer matches fixture")


if __name__=="__main__":
    main()
