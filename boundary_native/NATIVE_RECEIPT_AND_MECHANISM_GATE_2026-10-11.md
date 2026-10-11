# Native evidence receipt and component research gate — 2026-10-11

**Status:** Three genuine public-source defect/contrast pairs demonstrated across two Agent SDK families. A reusable *manually contract-grounded* typed-obligation validator passed its replay checks. **No new algorithm, automated scenario generator, baseline superiority or novel defect discovery proven.**

## Actual source-native evidence

| Defect family | Real SDK and issue | Old/fault outcome | Repaired/contrast outcome | What this does NOT prove |
|---|---|---|---|---|
| Guardrail exception accepted as successful output | OpenAI Agents SDK [#3280](https://github.com/openai/openai-agents-python/issues/3280), merged [#3281](https://github.com/openai/openai-agents-python/pull/3281) | Under pinned commit `683b6e79`, exception logged but not propagated; streamed final answer accepted | Under merged fix `4a3d33e`, RuntimeError reaches caller and final answer is not accepted. Normal control passes both. | It does not verify its separate concurrently cancelled guardrail sibling cleanup. |
| Approved tool output missing in streamed resumed state | OpenAI Agents SDK [#4125](https://github.com/openai/openai-agents-python/issues/4125), merged [#4126](https://github.com/openai/openai-agents-python/pull/4126) | Under `6e2095d`, streamed + guardrail tripwire stores function_call but no function_call_output; local tool executed once | Under `9d894a9`, call and output both persist. Normal run/stream controls match. | It does not prove any real remote external side effect, exactly-once behavior, or universal cross-mode trace equality. |
| Logical child task duplicate admission after retry | LangGraph [#8393](https://github.com/langchain-ai/langgraph/issues/8393), **open**. Community [PR #8485](https://github.com/langchain-ai/langgraph/pull/8485) **closed UNMERGED** | Exact official PR base `b2926a0` schedules new child despite same logical ID and an in-flight child | Community patch `e0d4ecf` reuses the pending child, no duplicate admission. Different-ID control schedules once in both. | No actual external effect was executed. Community fix is not an upstream merged fix. More recent main `b4991f1` still has object-vs-ID equality in source, but was NOT executed because its unpublished checkpoint dependency requirement could not be installed. |

Native Github Actions receipts:
- [#3280 action, both cases passed](https://github.com/MKLEE222/NLP/actions/runs/38114666257)
- [#4125 action, both cases passed](https://github.com/MKLEE222/NLP/actions/runs/38114794512)
- [#8393 action, both cases passed](https://github.com/MKLEE222/NLP/actions/runs/38115012574)
- [Six real observations evaluated by one typed checker](https://github.com/MKLEE222/NLP/actions/runs/38115131689): 3 FAIL old/source and 3 PASS fixed/control; 7 deterministic unit tests.

Provenance:
- `native_evidence_20261011.json`: six unmodified source-native JSON structures transcribed from the three above GitHub Actions job logs; recorded run/job IDs, full source SHAs, raw case outcomes and source author bug citations. The transcription is reviewable from upstream logs; it is not an independent SDK execution.
- `obligation_core.py`: three simple, explicitly source-grounded semantic checks (guardrail exception, approved tool result pairing, in-flight task ID); **ignores** the fixture's native expected PASS/FAIL labels. Returns PASS/FAIL/UNKNOWN.
- `test_obligation_core.py`: failures, fixes, unknown data, local tool effects, and output-status-only blindness. Unit tests on a committed native ledger are not evidence of an automatically synthesized oracle.
- Initial SDK #3280 CI failure was dependency drift, not the defect: the old SDK required its uv.lock openai==2.26.0. After that pin, source/fix results passed. LangGraph Oct 10 latest `main` failed *at package installation* because `langgraph-checkpoint>=4.3.0` was not available; using exact published community patch base `1.2.10` with resolvable dependencies fixed comparison scope without changing the code-path issue. Preserve both failures as provenance.

## Strong near-neighbor alert — important

[AgentCI](https://github.com/jinngimk-lang/agentci) already has provider-neutral upstream issue intake, provenance/case/trajectory fixture documentation and fail-closed evidence validation. It is a direct open-source engineering comparator for **generic event receipts**. Do not present the current hand-grounded `obligation_core.py` or six fixtures as a new full method. Mature property-based stateful testing, metamorphic testing (including ACL 2026 SGVEF-LOOP), message/schedule exploration (ISSTA 2024 DSTest tool demo), trace/model checking, linearizability and patch validation remain major scientific comparators.

**Our next method candidate, NOT established:** an automatic **source-constrained reachable failure-boundary synthesizer** that (1) discovers actual SDK execution and persistence transitions, (2) identifies legal fault points, (3) creates a minimal test pair with explicit negative controls and logical effect identities, and (4) emits obligations that can be checked independent of SDK-specific wrappers. The existing typed oracle would be its *evaluation backend*, not a paper-worthy contribution by itself.

## Mechanism comparison, not baseline death sentence

Candidate A: automatic reachable failure-boundary scenario generation. First priority. Seek measurable valid test coverage, previously unknown adjacent faults or cost savings relative to published/standard tools; ablate reachability, effect identity and contract oracle. No novelty claim before it can generate **unseen** native cases without hand-writing each complete fixture.

Candidate B: policy/effect-lineage partial-order oracle. Useful supporting component, but its standalone novelty is threatened by established runtime transactional testing, linearizability, and AgentCI. It can still be valuable as a strong, low-false-positive building block.

Candidate C: patch-sensitive pre/fix/negative-control witness generator. Independent direction but near neighbors include patch testing, mutation-guided repair, STING and other test-augmentation tools. A real advantage requires generalization beyond the existing author regression tests.

## Fixed method-gates

- **M0 reality:** 3 reproducible native historical cases from 2 frameworks. PASS for limited branch-specific observations. Distinct issue units: #3280, #4125, #8393.
- **M1 distinct mechanism:** automatic creation of a reachable, legal failure case and semantic test oracle on a source issue *not used to hand-code the component*. NOT PASSED (current implementation hand-grounded).
- **M2 comparable strong baselines:** original issue test, generic differential and metamorphic test, random fault injection and concurrency scheduling, existing AgentCI fixture checking. Same budget, source, state/authority and negative controls. NOT RUN.
- **M3 ablation:** isolate boundary reachability, effect correlation, obligation generation. NOT RUN.
- **M4 confirmation:** untouched issues across 2+ frameworks, no cherry-picking by observed results. NOT RUN.
- **M5 potential paper form:** new reusable algorithm/interface with measured incremental outcome OR credible new natural bug benchmark/measurement finding. Undecided.

A method can be new and useful **even when established baselines solve a subset of cases**. It is not scientific, though, merely to rename those baselines or to reconstruct existing manually written tests with more lines of code.

No paid model/API/GPU, no production writes, no private manuscript and no venue selected. Preserve historical commits and failed dependency runs.
