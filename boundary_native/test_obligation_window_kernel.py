"""Counterexample-oriented unit corpus for the *generic* obligation-window core.

Fixture snippets are tiny programs, not independent real-world bugs.
The naive comparator is a trivial token scanner, NOT FailMapper/FRIES/MR-Scout.
"""
from __future__ import annotations

import copy
import unittest

from obligation_window_kernel import (
    choose_by_obligation, compile_source, naive_token_baseline,
    verify_source_plan,
)


BASE="""\
import asyncio

async def client():
    queue=asyncio.Queue()
    async def writer(output_queue):
        x = await asyncio.gather(slow_tool())
        output_queue.put_nowait(None)
        return x
    job=asyncio.create_task(writer(queue))
    while True:
        token=await queue.get()
        if token is None:
            break
"""

PROTECTED="""\
import asyncio

async def client():
    queue=asyncio.Queue()
    async def writer(output_queue):
        try:
            x = await asyncio.gather(slow_tool())
        finally:
            output_queue.put_nowait(None)
        return x
    job=asyncio.create_task(writer(queue))
    while True:
        token=await queue.get()
        if token is None:
            break
"""

VARIANTS=[
    ("unguarded_gather",BASE,True),
    ("protected_finally",PROTECTED,False),
    ("different_writer_queue",BASE.replace("writer(queue)","writer(unrelated)"),False),
    ("different_consumer_queue",BASE.replace("await queue.get()","await other_queue.get()"),False),
    ("no_task_spawn",BASE.replace("asyncio.create_task(writer(queue))","writer(queue)"),False),
    ("no_queue_reader",BASE.replace("await queue.get()","await local_reader()"),False),
    ("different_producer_name",BASE.replace("writer","notify"),True),
    ("await_regular_function",BASE.replace("await asyncio.gather(slow_tool())",
                                           "await slow_tool()"),True),
    ("queue_keyword_binding",BASE.replace("writer(queue)","writer(output_queue=queue)"),True),
    ("normal_path_extra_event",BASE.replace(
        "output_queue.put_nowait(None)",
        "output_queue.put_nowait('payload')\n        output_queue.put_nowait(None)"),True),
    ("no_sentinel",BASE.replace("output_queue.put_nowait(None)","return None"),False),
    ("sentinel_other_queue",BASE.replace("output_queue.put_nowait(None)",
                                         "second_queue.put_nowait(None)"),False),
    ("immediate_return_before_await",BASE.replace(
        "x = await asyncio.gather(slow_tool())",
        "return 2\n        x = await asyncio.gather(slow_tool())"),True), # risk-only; unreachable not excluded by AST
]


class GenericWindowMethod(unittest.TestCase):
    def test_curated_counterexample_corpus(self):
        outputs=[]
        for name,src,expected in VARIANTS:
            with self.subTest(name=name):
                plan=compile_source(src,path=name+".py")
                got=plan["candidate_count"]>0
                self.assertEqual(got,expected)
                self.assertEqual(plan["status"],"SOURCE_TEST_PLAN_NOT_EXECUTED")
                verify_source_plan(src,plan)
                outputs.append((name,got,naive_token_baseline(src)))
        self.assertEqual(sum(x[1] for x in outputs),6)
        # The comparator is intentionally weak and includes wrong-queue cases.
        self.assertGreater(sum(x[2] for x in outputs),sum(x[1] for x in outputs))

    def test_no_circular_status_labels(self):
        plan=compile_source(BASE)
        self.assertEqual(plan["candidate_count"],1)
        case=plan["candidate_windows"][0]
        self.assertEqual(case["source_obligation"],"stream_terminal_delivery")
        self.assertEqual(case["static_status"],"TEST_PLAN_ONLY")
        self.assertEqual(case["queue_producer_name"],"output_queue")
        self.assertEqual(case["queue_consumer_name"],"queue")
        self.assertEqual(case["fault"],"cancel_at_producer_await")
        self.assertEqual(case["normal_control"],"complete_producer_await")

    def test_protected_finally_has_no_fault_window(self):
        self.assertEqual(compile_source(PROTECTED)["candidate_count"],0)

    def test_plan_replay_rejects_rewritten_obligation(self):
        plan=compile_source(BASE)
        forged=copy.deepcopy(plan)
        forged["candidate_windows"][0]["required_runtime_check"]="claim_success"
        with self.assertRaisesRegex(ValueError,"not independently reproduced"):
            verify_source_plan(BASE,forged)

    def test_plan_replay_rejects_source_drift(self):
        plan=compile_source(BASE)
        with self.assertRaisesRegex(ValueError,"not independently reproduced"):
            verify_source_plan(BASE+"\n# changed\n",plan)

    def test_budget_selection_is_deterministic(self):
        src=BASE.replace("async def client():","async def client_a():")
        src+="\n"+BASE.replace("async def client():","async def client_b():")
        cases=compile_source(src)["candidate_windows"]
        self.assertGreaterEqual(len(cases),2)
        self.assertEqual(len(choose_by_obligation(cases,1)),1)
        self.assertEqual(len(choose_by_obligation(cases,2)),2)
        self.assertEqual(choose_by_obligation(cases,2),choose_by_obligation(cases,2))
        self.assertEqual(choose_by_obligation(cases,0),[])
        with self.assertRaises(ValueError):
            choose_by_obligation(cases,-1)

    def test_multiple_failing_awaits_share_one_obligation_budget(self):
        src=BASE.replace(
            "x = await asyncio.gather(slow_tool())",
            "first = await slow_tool()\n        x = await asyncio.gather(slow_tool())"
        )
        plan=compile_source(src)
        self.assertEqual(plan["candidate_count"],2)
        self.assertEqual(len(choose_by_obligation(plan["candidate_windows"],2)),1)


if __name__=="__main__":
    unittest.main()
