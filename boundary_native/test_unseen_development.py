"""Reproducible preregistration and bounded AST test synthesis unit checks."""
import json
import tempfile
import unittest
from pathlib import Path
from source_guided_synth import synthesize, verify_pinned_source

BASE=Path(__file__).parent
REGISTRY=json.loads((BASE/"unseen_issue_registry_v1.json").read_text())

SYNTHETIC_CANCEL="""\
import asyncio
async def consumer():
    stream=asyncio.Queue()
    async def child_producer(stream_queue):
        result = await asyncio.gather(slow_tool())
        stream.put_nowait(None)
        return result
    task = asyncio.create_task(child_producer(stream))
    while True:
        event = await stream.get()
        if event is None:
            break
"""

SYNTHETIC_CLEANUP="""\
import asyncio
async def consumer():
    stream=asyncio.Queue()
    async def child_producer(stream_queue):
        try:
            result = await asyncio.gather(slow_tool())
        finally:
            stream.put_nowait(None)
        return result
    task = asyncio.create_task(child_producer(stream))
    while True:
        event = await stream.get()
        if event is None:
            break
"""


def fnv32(text):
    h=2166136261
    for char in text:
        h=((h ^ ord(char)) * 16777619) & 0xffffffff
    return h


class FrozenUnseenRegistry(unittest.TestCase):
    def test_only_four_dev_issue_bodies_may_be_read(self):
        issues=REGISTRY["cases"]
        self.assertEqual(len(issues),16)
        self.assertEqual(sum(x["role"]=="development" for x in issues),4)
        self.assertEqual(sum(x["role"]=="sealed_confirmation" for x in issues),12)
        self.assertEqual(len(set((x["repo"],x["number"]) for x in issues)),16)
        self.assertEqual(len(set(x["repo"] for x in issues)),4)

    def test_original_hash_split_is_reproducible(self):
        seed=REGISTRY["ranking"]["seed"]
        groups={}
        for row in REGISTRY["cases"]:
            score=fnv32(seed+"|"+row["repo"]+"#"+str(row["number"]))
            self.assertEqual(score,row["selection_score"])
            groups.setdefault(row["repo"],[]).append(row)
        for rows in groups.values():
            smallest=min(rows,key=lambda x:(x["selection_score"],x["number"]))
            self.assertEqual(smallest["role"],"development")
            for row in rows:
                if row is not smallest:
                    self.assertEqual(row["role"],"sealed_confirmation")


class SourceSentinelCompiler(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.file=self.root / "python/packages/autogen-agentchat/src/autogen_agentchat/agents/_assistant_agent.py"
        self.file.parent.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_finally_source_yields_bounded_cancel_scenario(self):
        self.file.write_text(SYNTHETIC_CANCEL)
        plan=synthesize(self.root,"autogen-tool-cancellation")
        self.assertEqual(plan["runnable_count"],1)
        self.assertEqual(plan["selected"][0]["producer"],"child_producer")
        self.assertEqual(plan["selected"][0]["injection"]["variants"],
                         ["normal_complete","cancel_inflight"])
        verify_pinned_source(self.root,plan)

    def test_finally_cleanup_excludes_false_source_candidate(self):
        self.file.write_text(SYNTHETIC_CLEANUP)
        plan=synthesize(self.root,"autogen-tool-cancellation")
        self.assertEqual(plan["runnable_count"],0)
        self.assertEqual(plan["selected"],[])

    def test_no_stream_consumer_is_not_sufficient(self):
        self.file.write_text(SYNTHETIC_CANCEL.replace(
            "event = await stream.get()","event = None"))
        plan=synthesize(self.root,"autogen-tool-cancellation")
        self.assertEqual(plan["runnable_count"],0)

    def test_no_scheduled_task_is_not_sufficient(self):
        self.file.write_text(SYNTHETIC_CANCEL.replace(
            "task = asyncio.create_task(child_producer(stream))",
            "task = child_producer(stream)"))
        plan=synthesize(self.root,"autogen-tool-cancellation")
        self.assertEqual(plan["runnable_count"],0)

    def test_unrelated_queue_consumer_is_not_a_positive(self):
        # The producer sends the terminal to a different queue from the
        # one being awaited. A broad pattern scanner would misclassify.
        src=SYNTHETIC_CANCEL.replace(
            "task = asyncio.create_task(child_producer(stream))",
            "other = asyncio.Queue()\n    task = asyncio.create_task(child_producer(other))")
        self.file.write_text(src)
        plan=synthesize(self.root,"autogen-tool-cancellation")
        self.assertEqual(plan["runnable_count"],0)

    def test_queue_passed_by_keyword_is_linked(self):
        src=SYNTHETIC_CANCEL.replace(
            "child_producer(stream)","child_producer(stream_queue=stream)")
        self.file.write_text(src)
        plan=synthesize(self.root,"autogen-tool-cancellation")
        self.assertEqual(plan["runnable_count"],1)
        self.assertEqual(plan["selected"][0]["queue_flow"]["consumer_queue"],"stream")

    def test_source_drift_rejected_after_plan_created(self):
        self.file.write_text(SYNTHETIC_CANCEL)
        plan=synthesize(self.root,"autogen-tool-cancellation")
        self.file.write_text(SYNTHETIC_CANCEL+"\n# unrelated source drift\n")
        with self.assertRaisesRegex(ValueError,"drifted"):
            verify_pinned_source(self.root,plan)


if __name__=="__main__":
    unittest.main()
