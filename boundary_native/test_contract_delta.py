"""Contract-gated increment, adversarial identity tests and classical comparisons.

All source snippets are DEVELOPMENT snippets invented before a blind test.
They are not new bugs or independent measurements of generalization.
"""
from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from classical_cut_baselines import (
    method_summary, raw_sites, source_lineage_windows, source_local_windows,
    trace_guided_sites, uniform_sites,
)
from contract_delta import (
    judge_native, source_to_obligation_plans, validate_catalog,
    verify_contract_plan,
)
from obligation_window_kernel import compile_source
from test_obligation_window_kernel import BASE, PROTECTED, VARIANTS, LOGICAL_TASK

ROOT=Path(__file__).parent
CATALOG=json.loads((ROOT/"normative_contracts_development_v1.json").read_text())


def registry_for(source:str,kind="stream_terminal_delivery"):
    digest=hashlib.sha256(source.encode("utf-8")).hexdigest()
    return {
        "schema":"trusted_local_adapters_v1",
        "adapters":{kind:{"local_only":True,"source_sha256":[digest],
                          "supported_callers":(["_call","_acall_impl"] if kind==
                            "inflight_logical_task_not_reissued" else ["client"])}},
    }


def pair(identity="run-key"):
    normal={"observation_origin":"native","authorized":True,
            "control_valid":True,"identity":identity}
    fault={"observation_origin":"native","authorized":True,
           "fault_triggered":True,"identity":identity}
    return normal,fault


class SourcePlanTests(unittest.TestCase):
    def test_catalog_is_externally_scoped_and_local_only(self):
        validate_catalog(CATALOG)
        self.assertEqual(len(CATALOG["obligations"]),4)
        bad=copy.deepcopy(CATALOG)
        bad["obligations"]["stream_terminal_delivery"]["side_effect_permission"]="remote_write"
        with self.assertRaisesRegex(ValueError,"non-local effects"):
            validate_catalog(bad)

    def test_ready_coverage_without_issue_selector(self):
        adapters=registry_for(BASE)
        m=source_to_obligation_plans(
            BASE,catalog=CATALOG,supported_adapters=adapters,budget=1,
        )
        self.assertEqual(m["source_candidate_count"],1)
        self.assertEqual(m["source_supported_count"],1)
        self.assertEqual(m["source_unknown_count"],0)
        self.assertEqual(len(m["selected"]),1)
        self.assertEqual(m["estimated_native_pair_attempts"],2)
        self.assertEqual(m["examined"][0]["identity_certificate"]["status"],
                         "PROVED_STRUCTURAL")
        verify_contract_plan(BASE,m,catalog=CATALOG,supported_adapters=adapters)

    def test_unknown_not_dropped_when_native_adapter_missing(self):
        m=source_to_obligation_plans(
            BASE,catalog=CATALOG,
            supported_adapters={"schema":"trusted_local_adapters_v1","adapters":{}},
        )
        self.assertEqual(m["source_candidate_count"],1)
        self.assertEqual(m["source_unknown_count"],1)
        self.assertEqual(m["selected"],[])
        self.assertEqual(m["examined"][0]["reason"],"trusted_native_adapter_unavailable")

    def test_unknown_for_adapter_of_wrong_source_sha(self):
        m=source_to_obligation_plans(
            BASE+"#source_changed\n",catalog=CATALOG,
            supported_adapters=registry_for(BASE),
        )
        self.assertEqual(m["source_candidate_count"],1)
        self.assertEqual(m["source_unknown_count"],1)
        self.assertEqual(m["examined"][0]["reason"],
                         "adapter_not_authorized_for_exact_source")

    def test_not_legal_fault_variant_is_unknown(self):
        catalog=copy.deepcopy(CATALOG)
        catalog["obligations"]["stream_terminal_delivery"]["legal_fault_variants"]=[
            "unrelated_fault"
        ]
        m=source_to_obligation_plans(
            BASE,catalog=catalog,supported_adapters=registry_for(BASE),
        )
        self.assertEqual(m["source_unknown_count"],1)
        self.assertEqual(m["selected"],[])
        self.assertEqual(m["examined"][0]["reason"],
                         "fault_not_licensed_by_normative_contract")

    def test_tampered_runtime_authority_or_plan_is_rejected(self):
        adapters=registry_for(BASE)
        plan=source_to_obligation_plans(
            BASE,catalog=CATALOG,supported_adapters=adapters
        )
        forged=copy.deepcopy(plan)
        forged["examined"][0]["fault"]="skip_identity_check"
        with self.assertRaisesRegex(ValueError,"plan changed"):
            verify_contract_plan(BASE,forged,catalog=CATALOG,
                                 supported_adapters=adapters)
        modified_catalog=copy.deepcopy(CATALOG)
        modified_catalog["obligations"]["stream_terminal_delivery"]["authority_scope"]+=" CHANGED"
        with self.assertRaisesRegex(ValueError,"plan changed"):
            verify_contract_plan(BASE,plan,catalog=modified_catalog,
                                 supported_adapters=adapters)

    def test_guaranteed_finally_has_no_candidate(self):
        m=source_to_obligation_plans(
            PROTECTED,catalog=CATALOG,
            supported_adapters=registry_for(PROTECTED),
        )
        self.assertEqual(m["source_candidate_count"],0)

    def test_same_task_obligation_can_be_planned_on_both_sync_async(self):
        m=source_to_obligation_plans(
            LOGICAL_TASK,catalog=CATALOG,
            supported_adapters=registry_for(
                LOGICAL_TASK,"inflight_logical_task_not_reissued"),
            budget=2,
        )
        self.assertEqual(m["source_candidate_count"],2)
        self.assertEqual(m["source_supported_count"],2)
        self.assertEqual(len(m["selected"]),2)
        self.assertTrue(all(item["identity_certificate"]["identity_type"]
                            =="inflight_logical_task_id"
                            for item in m["examined"]))

    def test_realistic_adapter_scope_marks_async_source_site_unknown(self):
        adapters=registry_for(LOGICAL_TASK,"inflight_logical_task_not_reissued")
        adapters["adapters"]["inflight_logical_task_not_reissued"]["supported_callers"]=["_call"]
        p=source_to_obligation_plans(
            LOGICAL_TASK,catalog=CATALOG,supported_adapters=adapters,budget=2)
        self.assertEqual(p["source_candidate_count"],2)
        self.assertEqual(p["source_supported_count"],1)
        self.assertEqual(p["source_unknown_count"],1)
        self.assertEqual(len(p["selected"]),1)
        self.assertEqual(p["examined"][0]["planning_status"],"UNKNOWN")
        self.assertEqual(p["examined"][0]["reason"],
                         "source_callable_not_executable_by_native_adapter")

    def test_no_external_rights_promoted_by_source_only(self):
        m=source_to_obligation_plans(
            BASE,catalog=CATALOG,supported_adapters={
                "schema":"trusted_local_adapters_v1",
                "adapters":{"stream_terminal_delivery":{
                    "local_only":False,
                    "source_sha256":[hashlib.sha256(BASE.encode()).hexdigest()],
                    "supported_callers":["client"],
                }},
            },
        )
        self.assertEqual(m["source_unknown_count"],1)


class NativeContractTests(unittest.TestCase):
    def test_native_stream_runtime_fail_pass_and_unknown(self):
        n,f=pair()
        n["terminal"]=True
        f.update(terminal=False,bounded_timeout=True)
        self.assertEqual(judge_native("stream_terminal_delivery",n,f,CATALOG).status,"FAIL")
        f.update(terminal=True,bounded_timeout=False)
        self.assertEqual(judge_native("stream_terminal_delivery",n,f,CATALOG).status,"PASS")
        f.pop("bounded_timeout")
        self.assertEqual(judge_native("stream_terminal_delivery",n,f,CATALOG).status,"UNKNOWN")

    def test_native_pending_task_identity(self):
        n,f=pair("child-22")
        n["independent_id_admissions"]=1
        f["original_child_pending"]=True
        f["same_id_extra_admissions"]=1
        self.assertEqual(judge_native("inflight_logical_task_not_reissued",n,f,CATALOG).status,"FAIL")
        f["same_id_extra_admissions"]=0
        self.assertEqual(judge_native("inflight_logical_task_not_reissued",n,f,CATALOG).status,"PASS")
        f["same_id_extra_admissions"]=None
        self.assertEqual(judge_native("inflight_logical_task_not_reissued",n,f,CATALOG).status,"UNKNOWN")

    def test_persisted_effect_exact_identity_not_just_count(self):
        n,f=pair("approved-tool")
        f["tool_execution_count"]=1
        f["committed_call_ids"]=["call-A"]
        f["persisted_result_ids"]=["call-B"]
        self.assertEqual(judge_native("approved_tool_result_persisted",n,f,CATALOG).status,"FAIL")
        # A count-only check would silently claim PASS in the case above.
        f["persisted_result_ids"]=["call-A"]
        self.assertEqual(judge_native("approved_tool_result_persisted",n,f,CATALOG).status,"PASS")
        f["persisted_result_ids"]=[]
        self.assertEqual(judge_native("approved_tool_result_persisted",n,f,CATALOG).status,"FAIL")
        f["persisted_result_ids"]=None
        self.assertEqual(judge_native("approved_tool_result_persisted",n,f,CATALOG).status,"UNKNOWN")

    def test_guardrail_failclosed(self):
        n,f=pair("streamed-output")
        n["accepted_final"]=True
        f.update(accepted_final=True,exception_propagated=False)
        self.assertEqual(judge_native("output_guardrail_fail_closed",n,f,CATALOG).status,"FAIL")
        f.update(accepted_final=False,exception_propagated=True)
        self.assertEqual(judge_native("output_guardrail_fail_closed",n,f,CATALOG).status,"PASS")

    def test_missing_authority_or_foreign_identity_is_unknown(self):
        n,f=pair()
        n["terminal"]=True
        f.update(terminal=False,bounded_timeout=True)
        f["identity"]="other-call"
        self.assertEqual(judge_native("stream_terminal_delivery",n,f,CATALOG).status,"UNKNOWN")
        f["identity"]=n["identity"]
        f["authorized"]=False
        self.assertEqual(judge_native("stream_terminal_delivery",n,f,CATALOG).status,"UNKNOWN")
        f["authorized"]=True
        f["observation_origin"]="synthetic"
        self.assertEqual(judge_native("stream_terminal_delivery",n,f,CATALOG).status,"UNKNOWN")


class ClassicalBaselineTests(unittest.TestCase):
    def test_same_source_observability_and_hash_stability(self):
        cases=raw_sites(BASE)
        self.assertGreaterEqual(len(cases),2)
        self.assertEqual(uniform_sites(BASE,1),uniform_sites(BASE,1))
        self.assertEqual(len(uniform_sites(BASE,1)),1)
        self.assertEqual(len(uniform_sites(BASE,2)),2)
        self.assertNotEqual(uniform_sites(BASE,1,seed="other"),[])

    def test_neither_local_nor_lineage_equals_contract_finally_check(self):
        self.assertGreater(len(source_local_windows(PROTECTED)),0)
        self.assertGreater(len(source_lineage_windows(PROTECTED)),0)
        self.assertEqual(compile_source(PROTECTED)["candidate_count"],0)

    def test_lineage_handles_unrelated_queue_better_than_local(self):
        wrong=BASE.replace("writer(queue)","writer(other_queue)")
        self.assertGreater(len(source_local_windows(wrong)),0)
        self.assertEqual(len(source_lineage_windows(wrong)),0)

    def test_true_trace_baseline_abstains_without_real_trace(self):
        m=trace_guided_sites(BASE,None,1)
        self.assertEqual(m["status"],"UNSUPPORTED_NO_REAL_HEALTHY_TRACE")
        self.assertEqual(m["selected"],[])
        positive=trace_guided_sites(BASE,{x["line"] for x in raw_sites(BASE)},2)
        self.assertEqual(positive["status"],"RANKED_FROM_REAL_HEALTHY_TRACE")
        self.assertGreaterEqual(positive["eligible_observed"],1)

    def test_development_structural_denominator_not_bugs(self):
        # All outcomes are SOURCE-ELIGIBILITY labels from hand-authored
        # micro examples. No real native conclusion is drawn here.
        rows=[]
        for label,source,expected in VARIANTS:
            rows.append((label,expected,
                         bool(source_local_windows(source)),
                         bool(source_lineage_windows(source)),
                         bool(compile_source(source)["candidate_windows"])))
        self.assertEqual(len(rows),15)
        self.assertEqual(sum(row[1] for row in rows),5)
        self.assertEqual(sum(row[4] for row in rows),5)
        self.assertGreaterEqual(sum(row[2] for row in rows),
                                sum(row[4] for row in rows))


if __name__=="__main__":
    unittest.main()
