"""Contract-conditioned obligation test synthesis and three-valued adjudication.

Increment over obligation_window_kernel.py:
  (1) Source candidate is joined to a *separate, externally sourced* contract,
      identity requirements, allowed local fault and observation interface.
  (2) An unsupported or unauthorized join returns UNKNOWN instead of silently
      dropping a candidate or optimistically counting it as covered.
  (3) The same normative contract emits normal/fault actions AND validates real
      native observations after execution, never from the source pattern.
  (4) Source candidates are ranked by obligation coverage and cut-frontier
      proximity within an equal *number of fault/control attempts* budget.

This is a developer/engineering research prototype, not a new temporal-logic
theorem or a proof of improvement over LDFI, Sieve, FailMapper or property
testing. Source-derived risk is NOT a native bug verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from obligation_window_kernel import compile_source, verify_source_plan


KINDS = (
    "stream_terminal_delivery",
    "inflight_logical_task_not_reissued",
    "approved_tool_result_persisted",
    "output_guardrail_fail_closed",
)
STATUSES = ("PASS", "FAIL", "UNKNOWN")


@dataclass(frozen=True)
class Judgment:
    status: str
    reason: str
    type: str
    source: str = "NATIVE_OBSERVATION"
    def to_dict(self):
        return asdict(self)


def _unknown(kind: str, reason: str) -> Judgment:
    return Judgment("UNKNOWN", reason, kind)


def _count(value: Any) -> int | None:
    if type(value) is not int or value < 0:
        return None
    return value


def validate_catalog(doc: dict) -> None:
    if not isinstance(doc, dict) or doc.get("schema") != "normative_obligations_v1":
        raise ValueError("unsupported normative contract schema")
    entries=doc.get("obligations")
    if not isinstance(entries, dict):
        raise ValueError("missing obligation definitions")
    for kind, c in entries.items():
        if kind not in KINDS:
            raise ValueError("unsupported obligation kind")
        if not isinstance(c, dict) or not isinstance(c.get("authority_url"), str):
            raise ValueError(f"missing external normative authority: {kind}")
        if not c["authority_url"].startswith("https://") or not c.get("authority_scope"):
            raise ValueError(f"invalid authority/scope: {kind}")
        if c.get("side_effect_permission") != "local_only":
            raise ValueError(f"non-local effects disallowed: {kind}")
        modes=c.get("legal_fault_variants")
        if not isinstance(modes, list) or not modes or any(
            not isinstance(v,str) or not v for v in modes
        ):
            raise ValueError(f"missing permitted fault modes: {kind}")
        req=c.get("required_runtime_fields")
        if not isinstance(req, list) or not req or any(not isinstance(v,str) for v in req):
            raise ValueError(f"missing runtime evidence schema: {kind}")


def identity_certificate(window: dict) -> dict:
    kind=window.get("source_obligation")
    if kind == "stream_terminal_delivery":
        fields=("queue_producer_name","queue_consumer_name","spawned_at_line",
                "await_at_line","discharge_at_line")
        if not all(window.get(x) is not None for x in fields):
            return {"status":"UNKNOWN","reason":"missing_source_queue_identity"}
        if not (window["spawned_at_line"] >= 0
                and window["await_at_line"] < window["discharge_at_line"]):
            return {"status":"UNKNOWN","reason":"invalid_source_cut_order"}
        return {
            "status":"PROVED_STRUCTURAL",
            "identity_type":"queue_spawner_argument",
            "identity":"%s::%s" % (window["caller"],window["queue_consumer_name"]),
            "source_flow":"%s -> %s" % (
                window["queue_producer_name"],window["queue_consumer_name"]),
            "binding_limit":"lexically tracked; dynamic alias proof only after execution",
        }
    if kind == "inflight_logical_task_not_reissued":
        if not (
            window.get("existing_projection") in ("id","object") and
            isinstance(window.get("stored_task_identifier"),str) and
            isinstance(window.get("attempted_identifier"),str) and
            window["attempted_identifier"].endswith(".id")
        ):
            return {"status":"UNKNOWN","reason":"no_tracked_logical_id_relation"}
        return {
            "status":"PROVED_STRUCTURAL",
            "identity_type":"inflight_logical_task_id",
            "identity":"%s::%s" % (
                window["caller"],window["stored_task_identifier"]),
            "source_projection":window["existing_projection"],
            "binding_limit":"does not prove equality semantics for user-defined task classes",
        }
    return {"status":"UNKNOWN","reason":"no_source_identity_compiler_for_type"}


def source_to_obligation_plans(
    source: str, *, catalog: dict, supported_adapters: dict[str, dict],
    budget: int=2, path: str="<source>"
) -> dict:
    if budget < 0:
        raise ValueError("execution budget must be nonnegative")
    validate_catalog(catalog)
    if supported_adapters.get("schema")!="trusted_local_adapters_v1":
        raise ValueError("unsupported native adapter registry")
    native_source=compile_source(source,path=path)
    verify_source_plan(source,native_source)
    entries=catalog["obligations"]
    examined=[]
    for idx,cut in enumerate(native_source["candidate_windows"]):
        kind=cut["source_obligation"]
        cert=identity_certificate(cut)
        norm=entries.get(kind)
        source_line=cut.get("await_at_line",cut.get("guard_at_line"))
        unique="%s:%s:%s:%s" % (kind,cut["caller"],source_line,idx)
        static_status="READY"
        reason="contract_and_identity_have_source_support"
        if norm is None:
            static_status,reason="UNKNOWN","external_contract_not_registered"
        elif cert["status"]!="PROVED_STRUCTURAL":
            static_status,reason="UNKNOWN",cert["reason"]
        elif not isinstance(supported_adapters.get("adapters",{}).get(kind),dict):
            static_status,reason="UNKNOWN","trusted_native_adapter_unavailable"
        elif not (supported_adapters["adapters"][kind].get("local_only") is True
                  and hashlib.sha256(source.encode("utf-8")).hexdigest()
                  in supported_adapters["adapters"][kind].get("source_sha256", [])):
            static_status,reason="UNKNOWN","adapter_not_authorized_for_exact_source"
        elif cut["fault"] not in norm["legal_fault_variants"]:
            static_status,reason="UNKNOWN","fault_not_licensed_by_normative_contract"
        # The higher-order adapter MUST still check actual admission/rights.
        observation_fields=norm["required_runtime_fields"] if norm else []
        examined.append({
            "candidate_id":unique,
            "kind":kind,
            "source_line":source_line,
            "source_window":cut,
            "identity_certificate":cert,
            "contract_authority":norm["authority_url"] if norm else None,
            "normal":cut.get("normal_control"),
            "fault":cut.get("fault"),
            "required_observations":observation_fields,
            "planning_status":static_status,
            "reason":reason,
            "estimated_native_attempts":2,
            "candidate_ordinal":idx,
        })

    def deterministic_key(t):
        w=t["source_window"]
        anchor=w.get("discharge_at_line",w.get("guard_at_line",0))
        cut=t["source_line"]
        # Prefer a late interruption before required discharge, reducing
        # collateral work; source order is a heuristic, NOT a probability.
        distance=max(0,anchor-cut)
        return (distance,t["candidate_id"])

    qualified=[t for t in examined if t["planning_status"]=="READY"]
    selected=[]
    covered=set()
    for t in sorted(qualified,key=deterministic_key):
        c=t["identity_certificate"]
        group=(t["kind"],c.get("identity"))
        if group in covered:
            continue
        if len(selected)>=budget:
            break
        selected.append(t)
        covered.add(group)

    return {
        "schema":"contract_conditioned_delta_v1",
        "source":path,
        "source_sha256":hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "catalog_sha256":hashlib.sha256(
            json.dumps(catalog,sort_keys=True).encode("utf-8")).hexdigest(),
        "adapter_registry_sha256":hashlib.sha256(
            json.dumps(supported_adapters,sort_keys=True).encode("utf-8")).hexdigest(),
        "source_candidate_count":len(examined),
        "source_supported_count":sum(x["planning_status"]=="READY" for x in examined),
        "source_unknown_count":sum(x["planning_status"]=="UNKNOWN" for x in examined),
        "examined":examined,
        "selected":[x["candidate_id"] for x in selected],
        "attempt_budget":budget,
        "estimated_native_pair_attempts":2*len(selected),
        "status":"PLANS_ONLY_NO_BUG_VERDICTS",
        "limits":"human-approved normative contract and manually trusted SDK native adapter",
    }


def verify_contract_plan(
    source: str, plan: dict, *, catalog: dict,
    supported_adapters: dict,
) -> None:
    """Recompute the entire manifest, including its source and policy hashes.

    Adapter authorization and contract definitions must not change between
    generation and execution. This is a reproducibility check, not a
    cryptographic signature: GitHub repository maintainers can edit code.
    """
    if not isinstance(plan,dict) or plan.get("schema")!="contract_conditioned_delta_v1":
        raise ValueError("unsupported plan")
    expected=source_to_obligation_plans(
        source, catalog=catalog, supported_adapters=supported_adapters,
        budget=plan.get("attempt_budget",-1),path=plan.get("source","<source>"),
    )
    if plan != expected:
        raise ValueError("contract plan changed after source/authority selection")


def _safe_bool(value):
    return value if type(value) is bool else None


def judge_native(kind: str, normal: dict, fault: dict, catalog: dict) -> Judgment:
    """Contract-independent of source version/expected FIX or BUG labels.

    Inputs are *normalized native SDK observation events*, not hypothetical
    source output. Normalize source SDK raw receipts separately and preserve
    original job IDs and source commits for reproducibility.
    """
    validate_catalog(catalog)
    if kind not in catalog["obligations"]:
        return _unknown(kind,"normative_contract_missing")
    if not isinstance(normal,dict) or not isinstance(fault,dict):
        return _unknown(kind,"native_control_or_fault_missing")
    if normal.get("observation_origin")!="native" or fault.get("observation_origin")!="native":
        return _unknown(kind,"non_native_or_unverified_input")
    if normal.get("identity") != fault.get("identity") or not normal.get("identity"):
        return _unknown(kind,"logical_identity_mismatch")
    if _safe_bool(normal.get("authorized")) is not True or _safe_bool(fault.get("authorized")) is not True:
        return _unknown(kind,"authorization_missing_or_denied")
    if _safe_bool(normal.get("control_valid")) is not True:
        return _unknown(kind,"normal_execution_control_invalid")
    if _safe_bool(fault.get("fault_triggered")) is not True:
        return _unknown(kind,"fault_execution_not_observed")

    if kind=="stream_terminal_delivery":
        if _safe_bool(normal.get("terminal")) is not True:
            return _unknown(kind,"healthy_stream_did_not_terminate")
        terminal=_safe_bool(fault.get("terminal"))
        timed_out=_safe_bool(fault.get("bounded_timeout"))
        if terminal is None or timed_out is None:
            return _unknown(kind,"bounded_liveness_evidence_missing")
        if timed_out and not terminal:
            return Judgment("FAIL","required_terminal_absent_under_bounded_cancel",kind)
        if terminal and not timed_out:
            return Judgment("PASS","stream_cancel_has_observed_terminal",kind)
        return _unknown(kind,"contradictory_terminal_and_timeout_observations")

    if kind=="inflight_logical_task_not_reissued":
        if _count(normal.get("independent_id_admissions")) != 1:
            return _unknown(kind,"distinct_logical_id_negative_control_invalid")
        if _safe_bool(fault.get("original_child_pending")) is not True:
            return _unknown(kind,"existing_child_not_proven_in_flight")
        extra=_count(fault.get("same_id_extra_admissions"))
        if extra is None:
            return _unknown(kind,"extra_admission_count_missing")
        return Judgment(
            "FAIL" if extra>0 else "PASS",
            "same_inflight_logical_id_readmitted" if extra else "pending_logical_task_reused",
            kind,
        )

    if kind=="approved_tool_result_persisted":
        # Event-level matching: count equality is insufficient (swapped IDs).
        # For a missing result, even an incomplete ID trace can reveal a
        # certain violation. For a claimed PASS, all IDs must be observed.
        calls=fault.get("committed_call_ids")
        results=fault.get("persisted_result_ids")
        if not isinstance(calls,list) or not isinstance(results,list):
            return _unknown(kind,"durable_call_result_receipts_unavailable")
        if any(not isinstance(x,str) or not x for x in calls+results):
            return _unknown(kind,"durable_result_identity_not_observed")
        if len(calls)!=len(set(calls)) or len(results)!=len(set(results)):
            return Judgment("FAIL","duplicate_durable_call_or_result_identity",kind)
        if not calls or _count(fault.get("tool_execution_count")) is None:
            return _unknown(kind,"approved_tool_execution_unobserved")
        if len(calls)!=fault["tool_execution_count"]:
            return _unknown(kind,"local_tool_effect_count_disagrees_with_session")
        if set(calls)!=set(results):
            return Judgment("FAIL","approved_call_has_missing_or_wrong_result_id",kind)
        return Judgment("PASS","approved_call_and_result_identity_match",kind)

    if kind=="output_guardrail_fail_closed":
        if _safe_bool(normal.get("accepted_final")) is not True:
            return _unknown(kind,"normal_accepted_final_control_invalid")
        accepted=_safe_bool(fault.get("accepted_final"))
        propagated=_safe_bool(fault.get("exception_propagated"))
        if accepted is None or propagated is None:
            return _unknown(kind,"guardrail_exception_observation_incomplete")
        if accepted:
            return Judgment("FAIL","failed_guardrail_output_accepted",kind)
        if propagated:
            return Judgment("PASS","guardrail_exception_propagated_and_final_rejected",kind)
        return _unknown(kind,"rejection_without_observed_exception_contract")

    return _unknown(kind,"no_native_oracle_for_kind")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--catalog",required=True)
    p.add_argument("--adapters",required=True)
    p.add_argument("--budget",type=int,default=2)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    source=Path(args.source).read_text(encoding="utf-8")
    catalog=json.loads(Path(args.catalog).read_text())
    adapters=json.loads(Path(args.adapters).read_text())
    plan=source_to_obligation_plans(
        source,catalog=catalog,supported_adapters=adapters,
        budget=args.budget,path=str(Path(args.source)),
    )
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(plan,sort_keys=True,indent=2)+"\n")
    print("CONTRACT_DELTA_PLAN",json.dumps({
        "source_candidates":plan["source_candidate_count"],
        "ready":plan["source_supported_count"],
        "unknown":plan["source_unknown_count"],
        "selected":len(plan["selected"]),
        "estimated_pair_attempts":plan["estimated_native_pair_attempts"],
    },sort_keys=True))


if __name__=="__main__":
    main()
