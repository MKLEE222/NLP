"""Development-only epistemic contract tests. No remote effects or writes."""
from __future__ import annotations
import unittest

from epistemic_receipt_fibers import (
    Receipt,all_worlds,infer_worlds,naive_log_only_retry,
    next_read_only_receipt_query,source_scoped_observation_value,
)


def E(key,source,fact,value,call="call-A"):
    return Receipt(key,call,source,fact,value)


class ReceiptSetTest(unittest.TestCase):
    def test_orphaned_session_does_not_imply_remote_commit_not_happened(self):
        evidence=[
            E("journal-call","session_journal","call_recorded",True),
            E("journal-missing","session_journal","result_recorded",False),
        ]
        fiber=infer_worlds("call-A",evidence)
        self.assertEqual(fiber.world_count,4)
        self.assertEqual(fiber.can_replay_without_external_duplication,"UNKNOWN")
        self.assertIn("effect_committed",fiber.mandatory_next_evidence)
        self.assertEqual(naive_log_only_retry(evidence,"call-A"),"SUGGEST_RETRY")
        self.assertEqual(
            next_read_only_receipt_query(fiber,["idempotency_receipt"])["fact"],
            "effect_committed",
        )

    def test_independent_local_effect_receipt_rules_out_unsafe_retry(self):
        e=[
            E("call","session_journal","call_recorded",True),
            E("result","session_journal","result_recorded",False),
            E("effect","local_effect_counter","effect_committed",True),
        ]
        result=infer_worlds("call-A",e)
        self.assertEqual(result.world_count,2)
        self.assertEqual(result.can_replay_without_external_duplication,
                         "UNSAFE_OR_UNAUTHORIZED")
        self.assertIn("read-only",result.semantic_scope)

    def test_safety_requires_both_approval_and_noncommit(self):
        e=[
            E("approval","approval_authority","approval_valid",True),
            E("no-effect","idempotency_receipt","effect_committed",False),
        ]
        p=infer_worlds("call-A",e)
        self.assertEqual(p.can_replay_without_external_duplication,
                         "CONDITIONALLY_SAFE_ONLY_IF_EXTERNAL_REPLAY_IS_PERMITTED")
        e.append(E("denied","approval_authority","approval_valid",False))
        bad=infer_worlds("call-A",e)
        self.assertEqual(bad.status,"UNKNOWN")
        self.assertEqual(bad.unknown_reason,"inconsistent_authoritative_receipts")

    def test_nonmatching_logical_call_id_never_joins(self):
        other=E("foreign","local_effect_counter","effect_committed",True,
                call="call-B")
        fiber=infer_worlds("call-A",[other])
        self.assertEqual(fiber.world_count,len(list(all_worlds())))
        self.assertEqual(fiber.accepted_receipt_count,0)

    def test_same_receipt_id_repeat_does_not_double_count(self):
        r=E("r1","session_journal","call_recorded",True)
        a=infer_worlds("call-A",[r])
        b=infer_worlds("call-A",[r,r])
        self.assertEqual(a.world_keys,b.world_keys)
        self.assertEqual(b.duplicate_receipt_count,1)
        self.assertEqual(b.accepted_receipt_count,1)

    def test_same_receipt_id_conflicts_are_unknown(self):
        a=E("same","session_journal","call_recorded",True)
        b=E("same","session_journal","call_recorded",False)
        r=infer_worlds("call-A",[a,b])
        self.assertEqual(r.status,"UNKNOWN")
        self.assertEqual(r.unknown_reason,"conflicting_duplicate_receipt")

    def test_unlicensed_log_source_cannot_assert_external_commit(self):
        e=E("forged","session_journal","effect_committed",False)
        p=infer_worlds("call-A",[e])
        self.assertEqual(p.status,"UNKNOWN")
        self.assertEqual(p.unknown_reason,"source_cannot_attest_requested_fact")

    def test_no_evidence_auto_authority_or_effect_reconciliation(self):
        p=infer_worlds("call-A",[])
        self.assertEqual(p.can_replay_without_external_duplication,"UNKNOWN")
        self.assertIn("approval_valid",p.mandatory_next_evidence)
        self.assertEqual(
            next_read_only_receipt_query(p,[])["status"],"ABSTAIN")
        self.assertEqual(
            next_read_only_receipt_query(p,["approval_authority"])["fact"],
            "approval_valid",
        )

    def test_effect_from_distinct_authoritative_source_shrinks_only(self):
        source=[
            E("a","session_journal","call_recorded",True),
            E("b","session_journal","result_recorded",False),
        ]
        before=infer_worlds("call-A",source)
        after=infer_worlds("call-A",source+[
            E("c","local_effect_counter","effect_committed",True),
        ])
        self.assertLessEqual(after.world_count,before.world_count)
        self.assertTrue(set(after.world_keys)<=set(before.world_keys))

    def test_unauthorized_or_mutating_query_must_abstain(self):
        e=E("effect","local_effect_counter","effect_committed",True)
        bad=Receipt(e.receipt_id,e.logical_call_id,e.source_kind,
                    e.fact,e.value,claimed_authorized=False)
        self.assertEqual(infer_worlds("call-A",[bad]).status,"UNKNOWN")
        self.assertEqual(
            infer_worlds("call-A",[e],
                         authority_mode="perform_remote_retry").status,
            "UNKNOWN",
        )

    def test_exact_identity_query_not_leaky(self):
        e=[
            E("one","local_effect_counter","effect_committed",True,call="x"),
            E("two","local_effect_counter","effect_committed",False,call="y"),
        ]
        self.assertIs(source_scoped_observation_value(e,"effect_committed","x"),True)
        self.assertIs(source_scoped_observation_value(e,"effect_committed","y"),False)
        self.assertIsNone(source_scoped_observation_value(e,"approval_valid","x"))


if __name__=="__main__":
    unittest.main()
