# Cold-start v2: source-guided failure-boundary research (11 Oct 2026)

Status: **engineering source-to-scenario prototype validated on known cases; scientific novelty and out-of-sample transfer NOT established.** This branch is derived from the Draft native-bug research branch, not from the archived QAEdit line. It does not change OACR, BRFP or any submitted manuscript.

## 1. What the previous method did NOT show

The earlier three historical bug scripts each contained hand-selected fault conditions, normal controls and success predicates. A separate typed checker on their six native outputs achieved 3 FAIL on buggy versions and 3 PASS on fixed/community-patched versions. This is reproducible, but circular as a claim of discovery. The source/code of all three upstream bugs was known before this generator was designed. Repetition of test executions and static fixtures does not increase the independent sample beyond three issue units.

Other important direct neighbors:
- OpenAI Agents SDK officially publishes ScriptedModel/testing utilities for deterministic fault injection, step accounting, tool/session/guardrail and streamed workflow tests: https://openai.github.io/openai-agents-python/testing/ . These are existing SDK capabilities, not our invention.
- ISSTA 2024 **FRIES** already automatically synthesizes interacting API fuzz targets using an API dependency graph and ownership-validity analysis, identifying substantial real new bugs: https://2024.issta.org/details/issta-2024-papers/91/FRIES-Fuzzing-Rust-Library-Interactions-via-Efficient-Ecosystem-Guided-Target-Genera . Generic source/API dependency guided test generation is NOT our novel principle.
- ISSTA 2024 **SAScope** combines real software fault taxonomy with metamorphic and differential testing of framework internals: https://doi.org/10.1145/3650212.3680398 . A cross-framework anomaly study is not automatically new.
- ACL 2026 **SGVEF-LOOP** includes coverage-guided agent metamorphic exploration: https://aclanthology.org/2026.acl-long.1224/ .
- Existing **AgentCI** open-source engineering project already validates provenance-rich provider-neutral issue fixtures: https://github.com/jinngimk-lang/agentci . A generic JSON receipt/checker is insufficient as the main contribution.

## 2. What we actually built *in this turn*

New research branch: research/source-guided-failure-synthesis-20261011.

- source_guided_synth.py: standard-library Python AST analysis of exact upstream SDK source files. Detects source-level (a) broad exception handling whose callback result flows to final accepted output, (b) resume-owned tool-result persistence ordering relative to a rejecting guardrail, (c) logical child-ID checks against already running tasks. A local static callgraph and source ordering constrain candidate generation; source hash, AST line and executable fault/negative-control variants are included in the immutable plan. Unrelated broad exception handlers and fresh-session-only persistence cases are excluded by explicit tests.
- execute_source_plan.py: small, separately audited, offline *native SDK adapters* implementing the emitted variants and observing actual final-output state, durable paired call IDs, child scheduling admissions. Includes normal guards, distinct child IDs and alternate execution modes as negative controls. No remote LLM or tool/network write.
- Input provenance is verified twice: fixed source sha256 and recompilation of selected AST candidates; forged contract/line selections fail closed. The compiler neither imports upstream Issue IDs nor reads an expected defect label. Source-derived hints are *not* used by the native runtime checker to decide PASS/FAIL.
- 20 static/unit regression checks on both previous and new components: syntax, exception recovery, approval/persistence source ordering, causal caller-to-final-output relationship, sync/async duplicate admission, missing data and forged plan rejection.
- Native GitHub Actions: two distinct SDK families, three known issue units, six pinned source/patch versions. The source compiler automatically generated *eight path-level tests* across those version pairs: one guardrail callback case for each version, one resumed approval case for each, **two LangGraph internal sync/async ID sites on each of two versions**. Each path tests at least one fault/control pair. Source-owned subcases are NOT eight independent issues.

## 3. Reproducible observed result, with denominators

CI: https://github.com/MKLEE222/NLP/actions/runs/38115822774 (source-generated real native SDK execution, all seven workflow jobs PASS; compiler controls include 20 tests).
Earlier first source-generated six-version run (LangGraph sync only): https://github.com/MKLEE222/NLP/actions/runs/38115660652 .
The persisted machine ledger is source_guided_native_evidence_20261011.json; its rows were transcribed directly from CI output with run/job ids, source hash, selected AST lines and raw native observations.

| Prior public issue (known, not discovered by us) | Original source observed effect | Fixed / community patched observed effect | Method interpretation |
|---|---|---|---|
| OpenAI Agents #3280 | Raised output guardrail is swallowed and final answer accepted. Structural handler is a broad exception returning empty list on a call path to final_output. | Exception propagated, no accepted final answer. AST plan still generates the *same* test. | One source-directed guardrail scenario, normal-control passed. |
| OpenAI Agents #4125 | After human approval the tool runs once, but streamed resumed guardrail trip saves call without paired output. AST inferred save-after-guardrail ordering. | Under merged fix, session persists call/output pair before output guardrail for approved resumed path. | Four-mode/guardrail combinations per version, one logical bug. |
| LangGraph #8393 | Under exact upstream patch-base revision, both _call and _acall_impl admit a duplicate same-logical-ID in-flight child. | Unmerged community patch replaces object-vs-ID with ID-vs-ID, both paths reuse pending child. Different-ID control schedules normally. | Two automatically selected source paths per version; **admission only**, no external tool effect. |

The six versions were **selected because their bugs/fixes are already known**. This validates a limited compiler/adapter, not discovery precision, recall, prevalence or superior test efficiency. Every native source test is on local fake tools; actual external effect durability is not established. The LangGraph patch is NOT merged upstream.

Static-search diagnostic only: the #3280 historical run_loop.py has eight broad exception handlers; the current bounded source-to-terminal chain/filter selects one compatible exception-boundary handler. This describes candidate pruning but is **not** a cost-matched baseline or a new fault discovery. Both before and after select the same candidate, avoiding post-hoc test switching.

## 4. Actual candidate mathematical problem (not a theorem)

Let P be a real SDK program, C externally authorized behavioral contracts, I the permissible injection points/faults, A the explicitly supported no-network execution adapters, and B an execution budget. A generator returns source-justified *plans*:

  G(P,C,I,A) = { (entrypoint, reachable control path, legal injected fault, normal control, observable downstream obligation) }.

For a plan t, real native execution returns D(t) in {FAIL,PASS,UNKNOWN}. Invalid or incomplete negative controls return UNKNOWN, not a bug/fix verdict. An eventual (NOT implemented) method may select tests by coverage of independently required obligations within a cost bound:

  maximize over S subset G, sum_{c in C} w_c [exists t in S that covers c], subject to sum_{t in S} cost(t) <= B.

This is a **standard coverage-constrained selection objective**, not a novel theoretical result. Our current program merely orders detected AST candidates and executes the first 1-2 under a fixed budget; it does not solve an optimization problem.

## 5. Cold-start deficiencies to fix before claiming a new component

1. **Adapter generalization:** the three compiler patterns and their exact native adapters were authored after reading the three real bugs. This is not issue-blind discovery, despite not providing issue IDs to the compiler. Need hold out complete bug families/framework implementations never inspected during method design.
2. **Static soundness:** AST reachability is overapproximate; function-name/callback relations and one caller-to-sink path do NOT prove the injected failure is dynamically reachable on a legal production path. Native execution currently establishes reachability only for selected profiles. False-positive and no-candidate counts must be measured over eligible source projects.
3. **Independent contracts:** required semantic obligations still need an external trusted specification. Current profiles already know which terminal effects matter. Automatically inferring safety policy from source is not justified, and some SDKs explicitly support different outcomes by mode.
4. **No baseline superiority:** no strong head-to-head comparison with published FRIES-style API graph fuzzing, standard stateful property testing/scheduling, SGVEF-LOOP metamorphic guidance, AgentCI receipts and each SDK's official test suite. A naive broad exception scan is merely a sanity baseline, NOT enough for a top venue claim.
5. **No unseen result:** all three units were previously reported. Whether the approach finds meaningful additional failures or improves valid fault-case generation is open. The historical 3 FAIL/3 PASS is a source compatibility proof only.
6. **Limited consequence access:** the test uses fake models and local tool callback effects. It does not prove native network/DB idempotency, true exactly-once delivery, cleanup, crash recovery or human approval semantics outside the selected branches.
7. **Dependency fidelity:** historical Python SDK release and key OpenAI type package versions were pinned and pip freeze captured; many transitive packages were not fully constrained by immutable lock hashes. A reproducible package environment remains a target.
8. **Statistical grain:** two sites in one LangGraph issue and four tests per approval case must not be treated as independent observed bugs. Aggregate by upstream issue/repository and include untouched negatives.
9. **CI selection:** before/fix labels enter the outer CI assertion but not the generator or runtime oracle. For true discovery experiments, remove any fix-specific expectations until after independently frozen outcomes.

## 6. Next high-value engineering/scientific arm

*Not yet started.* Freeze a naturally sampled, non-overlapping set of public runtime issues across at least two other ecosystems with deterministic/admissible local adapters. Reserve whole issue IDs and source commits **before** reading the held-out issues' fixes; avoid treating already inspected CrewAI #7252, LangGraph #6792 or OpenAI #4775 as untouched.

Build a generator for **source-to-executable failure schedules with contract projections**, where an adapter is generic per SDK or automatically composed from official scripted test interfaces, rather than handwritten for each discovered failure *class*. Compare **valid generated scenarios per setup hour, real fault finding at fixed budget, invalid-injection ratio, contract false alarms and median triage cost** against strongest runnable baselines. Ablate reachability, mode pairing, effect-ID tracking, and fail-closed UNKNOWN independently. Do not write a paper's novelty claim until this evidence exists.

The central research question remains open to both an original **new component** and a separate empirical benchmark paper. A strong existing comparator detecting some faults does NOT automatically refute either; merely catching historically known bugs also cannot establish them.
