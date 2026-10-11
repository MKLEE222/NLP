"""Public model-free QAEdit manifest regression checks; no model or dataset download."""
import copy
import importlib.util
import json
import random
import unittest
from pathlib import Path

MOD = Path(__file__).with_name("prepare_manifest.py")
spec = importlib.util.spec_from_file_location("qaedit_prepare_manifest", MOD)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def row(i):
    return {
        "Source": ("NQ", "TQA", "SQA")[i % 3],
        "prompt": "Question " + str(i) + " distinct?",
        "target": "Correct answer " + str(i),
        "subject": "Unique subject " + str(i),
        "rephrase": "Distinct paraphrase of question " + str(i) + "?",
        "locality": [{"loc": "Sentinel " + str(i) + "_" + str(k),
                      "loc_ans": "Safe " + str(k)} for k in range(5)],
    }


class ManifestChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [row(i) for i in range(12000)]
        cls.manifest = prep.build(cls.rows, "sha-fixture-UNREAL")

    def test_all_splits_have_no_observation_overlap(self):
        self.assertEqual(self.manifest["summary"]["cross_role_prompt_or_rephrase_leakage"], 0)
        self.assertEqual(self.manifest["summary"]["valid_unique_rows"], 12000)

    def test_role_groups_are_disjoint(self):
        pools, _ = prep.allocate(self.rows)
        sets = [set(pools.get(role, [])) for _, role in prep.BUCKETS]
        for i, first in enumerate(sets):
            for second in sets[i + 1:]:
                self.assertFalse(first.intersection(second))

    def test_histories_future_actions_and_sentinels_are_disjoint(self):
        for phase in ("dev", "confirm"):
            p = self.manifest[phase]
            history = [uid for pair in p["trajectories"]
                       for side in ("A", "B") for uid in pair[side]]
            future = [event["future_action_id"] for event in p["events"]]
            sentinels = p["sentinel_question_ids"]
            self.assertEqual(len(history + future + sentinels),
                             len(set(history + future + sentinels)))
            self.assertEqual(len(sentinels), 8)
            self.assertEqual(len(p["trajectories"]), prep.SCHEDULES[phase]["pairs"])
            self.assertEqual(len(p["events"]), p["candidate_state_pair_events"])

    def test_dataset_order_does_not_change_manifest(self):
        # Hash selection and connected-components must not depend on record order.
        reordered = list(reversed(self.rows))
        other = prep.build(reordered, "sha-fixture-UNREAL")
        self.assertEqual(json.dumps(other, sort_keys=True),
                         json.dumps(self.manifest, sort_keys=True))

    def test_connected_questions_cannot_leak(self):
        records = [row(i) for i in range(300)]
        records[0]["rephrase"] = records[1]["prompt"]
        pools, audit = prep.allocate(records)
        r0 = prep.clean_record(records[0])["uid"]
        r1 = prep.clean_record(records[1])["uid"]
        loc0 = next(role for role, ids in pools.items() if r0 in ids)
        loc1 = next(role for role, ids in pools.items() if r1 in ids)
        self.assertEqual(loc0, loc1)
        self.assertEqual(audit["cross_role_prompt_or_rephrase_leakage"], 0)

    def test_invalid_and_duplicate_rows_reported_not_repaired(self):
        records = [row(0), copy.deepcopy(row(0)), {"subject": "incomplete"}]
        _, audit = prep.allocate(records)
        self.assertEqual(audit["valid_unique_rows"], 1)
        self.assertEqual(audit["duplicate_rows"], 1)
        self.assertEqual(audit["invalid_rows"], 1)

    def test_insufficient_real_data_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            prep.build([row(i) for i in range(20)], "tiny-file")


if __name__ == "__main__":
    unittest.main()
