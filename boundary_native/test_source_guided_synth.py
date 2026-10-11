"""Adversarial model-free checks for the SOURCE planner, not native SDK claims."""
import tempfile
import unittest
from pathlib import Path

from source_guided_synth import synthesize, verify_pinned_source


AGENT_TEMPLATE = """
async def _run_output_guardrails_for_stream():
    try:
        return await run_output_guardrails()
    except Exception:
        logger.error("failure")
        ENDING

async def _finalize_streamed_final_output():
    result = await _run_output_guardrails_for_stream()
    streamed_result.final_output = result

async def unrelated_broad_exception():
    try:
        whatever()
    except Exception:
        return []

async def pretend_guardrail():
    try:
        return await run_output_guardrails()
    except Exception:
        return []
"""

ADMISSION_TEMPLATE = """
def _call():
    pending = futures()
    next_task = schedule_task()
    return next(
        (f for f,t in list(pending.items()) if t is not None and LEFT == next_task.id),
        None
    )

async def _acall():
    next_task = schedule_task()
    return next((f for f,t in list(futures().items()) if LEFT == next_task.id), None)
"""


class SourceCompilerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_source(self, profile, content):
        from source_guided_synth import SOURCE_PROFILES
        target=self.root / SOURCE_PROFILES[profile][0]
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(content)
        return target

    def test_agent_sink_reachability_filters_decoys(self):
        self.write_source("agents-guardrail",AGENT_TEMPLATE.replace("ENDING","return []"))
        plan=synthesize(self.root,"agents-guardrail")
        self.assertEqual(plan["candidate_count"],1)
        self.assertEqual(plan["runnable_count"],1)
        self.assertEqual(len(plan["selected"]),1)
        self.assertEqual(plan["selected"][0]["handler_exit"],"return_empty_list")
        self.assertEqual(plan["selected"][0]["sink_call_path"],
                         ["_finalize_streamed_final_output",
                          "_run_output_guardrails_for_stream"])
        self.assertGreater(plan["search_space"]["broad_exception_handler_count"],
                           len(plan["selected"]))
        verify_pinned_source(self.root,plan)

    def test_repaired_exception_is_same_generated_probe(self):
        self.write_source("agents-guardrail",AGENT_TEMPLATE.replace("ENDING","raise"))
        plan=synthesize(self.root,"agents-guardrail")
        self.assertEqual(plan["selected"][0]["handler_exit"],"explicit_reraise")
        self.assertEqual(plan["selected"][0]["injection"]["variants"],
                         ["pass","raise_RuntimeError"])

    def test_no_unsafe_promotion_of_irrelevant_exception(self):
        self.write_source("agents-guardrail",AGENT_TEMPLATE.replace(
            "ENDING","return []").replace(
                "result = await _run_output_guardrails_for_stream()","result = 1"))
        plan=synthesize(self.root,"agents-guardrail")
        self.assertFalse(plan["selected"])
        self.assertEqual(plan["runnable_count"],0)

    def test_langgraph_existing_object_v_id_and_same_controls(self):
        raw=ADMISSION_TEMPLATE.replace("pending.items()","futures().items()")
        self.write_source("langgraph-admission",raw.replace("LEFT","t"))
        plan=synthesize(self.root,"langgraph-admission")
        self.assertEqual(plan["candidate_count"],2)
        self.assertEqual(plan["runnable_count"],1)
        self.assertEqual(plan["selected"][0]["comparison"],
                         "stored_object_vs_candidate_id")
        self.assertFalse(plan["selected"][0]["is_async"])
        self.assertEqual(plan["selected"][0]["injection"]["variants"],
                         ["same_logical_id","different_logical_id"])

    def test_langgraph_fixed_id_also_runnable(self):
        raw=ADMISSION_TEMPLATE.replace("pending.items()","futures().items()")
        self.write_source("langgraph-admission",raw.replace("LEFT","t.id"))
        plan=synthesize(self.root,"langgraph-admission")
        self.assertEqual(plan["selected"][0]["comparison"],
                         "stored_id_vs_candidate_id")


    def test_resumed_approval_persistence_before_and_after_guardrail(self):
        old = """
async def finalize(save_items):
    result = await _run_output_guardrails_for_stream()
    streamed_result.final_output = result
    await save_items(items)

async def resumed():
    await finalize(save_items=_save_resumed_items)

async def fresh():
    await finalize(save_items=_save_fresh_items)
"""
        self.write_source("agents-approved-persistence",old)
        p=synthesize(self.root,"agents-approved-persistence")
        self.assertEqual(p["candidate_count"],1)
        self.assertEqual(p["selected"][0]["source_inferred_order"],
                         "resumed_guardrail_may_precede_save")
        self.assertEqual(p["selected"][0]["resumed_call_sites"],1)
        fixed = """
async def finalize(save_items,persist_before_output_guardrails):
    if persist_before_output_guardrails:
        await save_items(items)
    result = await _run_output_guardrails_for_stream()
    streamed_result.final_output = result
    if not persist_before_output_guardrails:
        await save_items(items)

async def resumed():
    await finalize(save_items=_save_resumed_items,
                   persist_before_output_guardrails=True)

async def fresh():
    await finalize(save_items=_save_fresh_items,
                   persist_before_output_guardrails=False)
"""
        self.write_source("agents-approved-persistence",fixed)
        q=synthesize(self.root,"agents-approved-persistence")
        self.assertEqual(q["candidate_count"],1)
        self.assertEqual(q["selected"][0]["source_inferred_order"],
                         "resumed_save_before_guardrail")
        self.assertEqual(q["selected"][0]["injection"]["variants"],
                         ["run_pass","streamed_pass","run_trip","streamed_trip"])

    def test_no_resumed_context_no_approval_obligation(self):
        self.write_source("agents-approved-persistence","""
async def finalize(save_items):
    result = await _run_output_guardrails_for_stream()
    streamed_result.final_output = result
    await save_items(items)
async def first_turn():
    await finalize(save_items=_save_fresh_items)
""")
        p=synthesize(self.root,"agents-approved-persistence")
        self.assertEqual(p["candidate_count"],0)

    def test_output_bound_to_exact_source(self):
        path=self.write_source("agents-guardrail",AGENT_TEMPLATE.replace("ENDING","raise"))
        plan=synthesize(self.root,"agents-guardrail")
        path.write_text(path.read_text()+"\n# source changed\n")
        with self.assertRaisesRegex(ValueError,"drifted"):
            verify_pinned_source(self.root,plan)

    def test_fails_closed_missing_source(self):
        with self.assertRaisesRegex(ValueError,"not in pinned"):
            synthesize(self.root,"agents-guardrail")

    def test_rejects_unknown_profile_and_zero_budget(self):
        with self.assertRaises(ValueError):
            synthesize(self.root,"unknown")
        with self.assertRaises(ValueError):
            synthesize(self.root,"agents-guardrail",budget=0)


if __name__=="__main__":
    unittest.main()
