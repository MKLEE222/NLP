"""Regression tests for a reusable but not-yet-original obligation prototype."""
import copy
import json
import unittest
from pathlib import Path
from obligation_core import PASS, FAIL, UNKNOWN, check_guardrail_exception, check_approved_result_pair, check_inflight_dedup, judge

BASE=Path(__file__).parent
CASES=json.loads((BASE/"native_evidence_20261011.json").read_text())["rows"]

class ObligationChecks(unittest.TestCase):
    def test_six_grounded_pre_post_provenance(self):
        self.assertEqual(len(CASES),6)
        results=[judge(r) for r in CASES]
        self.assertEqual(sum(r.status==FAIL for r in results),3)
        self.assertEqual(sum(r.status==PASS for r in results),3)
        self.assertTrue(all(r.status!=UNKNOWN for r in results))
        seen={}
        for row in CASES:
            seen.setdefault(row["channel"],[]).append(judge(row).status)
            self.assertTrue(row["job"] and row["run"])
        self.assertEqual(set(seen), {"guardrail_exception","approved_result","inflight_dedup"})
        for channel, verdicts in seen.items():
            self.assertCountEqual(verdicts,[PASS,FAIL],channel)

    def test_oracle_ignores_existing_fixture_verdict(self):
        for row in CASES:
            m=copy.deepcopy(row)
            m["observation"]["validation"]="FAKE_SUCCESS_LABEL"
            m["observation"]["verdict"]="FAKE_SUCCESS_LABEL"
            self.assertEqual(judge(row).status,judge(m).status)

    def test_unknown_is_not_success(self):
        for row in CASES:
            malformed={"channel":row["channel"],"observation":{}}
            self.assertEqual(judge(malformed).status,UNKNOWN)

    def test_guardrail_no_acceptance_under_injected_exception(self):
        old=next(x for x in CASES if x["channel"]=="guardrail_exception"
                 and x["observation"].get("validation")=="EXPECTED_HISTORICAL_FAIL_OPEN")
        x=copy.deepcopy(old["observation"])
        self.assertEqual(check_guardrail_exception(x).status,FAIL)
        x["fault"]["final_output_accepted"]=False
        x["fault"]["caught_exception"]="RuntimeError"
        self.assertEqual(check_guardrail_exception(x).status,PASS)

    def test_stream_status_alone_is_not_effect_pair_evidence(self):
        old=next(x for x in CASES if x["channel"]=="approved_result"
                 and x["observation"]["validation"]=="EXPECTED_HISTORICAL_STREAM_ORPHAN")
        future=next(x for x in CASES if x["channel"]=="approved_result"
                    and x["observation"]["validation"]=="EXPECTED_FIXED_APPROVED_EFFECT_PAIR")
        self.assertEqual(
            {(x["mode"],x["tripwire"]):x["status"] for x in old["observation"]["cases"]},
            {(x["mode"],x["tripwire"]):x["status"] for x in future["observation"]["cases"]}
        )
        self.assertEqual(check_approved_result_pair(old["observation"]).status, FAIL)
        self.assertEqual(check_approved_result_pair(future["observation"]).status, PASS)

    def test_logical_id_not_object_identity(self):
        old=next(x for x in CASES if x["channel"]=="inflight_dedup"
                 and x["observation"]["verdict"]=="EXPECTED_DUPLICATE_ADMISSION")
        fix=next(x for x in CASES if x["channel"]=="inflight_dedup"
                 and x["observation"]["verdict"]=="EXPECTED_DEDUP_ADMISSION")
        self.assertEqual(check_inflight_dedup(old["observation"]).status,FAIL)
        self.assertEqual(check_inflight_dedup(fix["observation"]).status,PASS)

    def test_no_claim_of_remote_exactly_once(self):
        fixed=next(x for x in CASES if x["channel"]=="approved_result"
                   and x["observation"]["validation"]=="EXPECTED_FIXED_APPROVED_EFFECT_PAIR")
        m=copy.deepcopy(fixed["observation"])
        # The local fixture cannot represent an independent provider receipt:
        # ambiguous remote write effects must not be upgraded to PASS.
        for row in m["cases"]:
            row["tool_executions"]=None
        self.assertEqual(check_approved_result_pair(m).status,UNKNOWN)

if __name__=="__main__":
    unittest.main()
