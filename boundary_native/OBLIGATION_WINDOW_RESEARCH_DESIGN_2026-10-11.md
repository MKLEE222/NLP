# Obligation-Window Research Design — after studying TOSEM, FSE, ASE, ISSTA and TSE standards

Date: 2026-10-11. Research status: executable V0 and a falsifiable *own-method hypothesis*, not independent novelty or paper-grade validation. This branch inherits the already frozen 16-issue registry; 12 confirmation issue bodies, code fixes, and outcomes stay sealed. Prior OACR, BRFP and neural-identifiability projects are conceptually separate.

## 1. Strong real prior work — technical content rather than titles

| Study | Accepted venue / distinctive actual mechanism | What that means for us |
|---|---|---|
| FRIES (Yin et al., 2024) | ISSTA main, weighted Rust API dependency graph + ownership-valid API-sequence fuzz-target generation; 130 bugs, 84 previously unknown and 54 confirmed | Source dependency-guided test generation and resource legality are already established. |
| FailMapper (Dong et al., 2025) | ASE main, nine failure scenarios + Monte Carlo tree search for bug-triggering unit tests. Reports 40/50 Defects4J bugs and 36 new bugs. Its public Java port uses JavaParser, LLM expansions and runtime verification. | Fault-scenario search or fault classification alone is not original. |
| DynaMOSA (Panichella, Kifetew & Tonella, 2018) | IEEE TSE, dynamic many-objective test generation prioritizing targets by control dependencies; 346 Java classes, mutation/branch coverage comparisons | Even dynamic budgeted control-dependency-aware test selection predates our method; current greedy source-window selector is plainly classical. |
| Valg (Kim et al., 2025) | ASE main, Distinguished Paper; selective runtime monitoring via a two-armed bandit optimizing redundant monitor checks | Budget-aware test/monitor selection is not new in isolation. |
| MR-Scout (Xu et al., 2024) | TOSEM, automatically mines and codifies metamorphic relations from developer-written tests, then filters weak relations | Automatic extraction of oracles from existing tests has a very strong journal prior. |
| MR-Coupler (Xu et al., 2026) | FSE main, derives function coupling from source and uses LLM-synthesized tests + amplification/mutation validation | Even source-driven cross-function test relation inference has been done. |
| CASCADE (Kiecker et al., 2026) | FSE main, tests from docs and independently generated reference behavior; only flags when original code fails AND document-derived code passes | Two-sided causal/control corroboration is not ours to claim as a new generic idea. |
| DepState (Fang et al., 2025) | ISSTA main, builds distributed DB table dependencies and varies cluster states to trigger sync defects; 22 new bugs, 11 confirmed | Strong domain-specific state/fault dependency testing already exists. |
| Onweer (Coremans/De Roover, 2025) | ICSME main, jointly fuzzes microservice workload and injected faults to reduce manual scenario design | General automated fault-injection schedule exploration is already mature. |
| Xue et al., 2025 | ASE Industry Showcase, taxonomy from 1,026 real bug instances in LLM workflow frameworks | We cannot claim to be first to study structural agent runtime bugs. |
| Yu et al., 2026 | TOSEM, a survey of failure analysis/fault injection in AI systems across many layers | Must demonstrate our exact mechanism is not a renamed failure-injection policy. |

References (accepted tracks/source documentation):
- FRIES https://doi.org/10.1145/3650212.3680348
- FailMapper https://conf.researchr.org/details/ase-2025/ase-2025-papers/212/FailMapper-Automated-Generation-of-Unit-Tests-Guided-by-Failure-Scenarios
- public FailMapper artifact https://github.com/nuwaLab/FailMapper
- DynaMOSA (TSE) https://doi.org/10.1109/TSE.2017.2663435
- Valg https://conf.researchr.org/details/ase-2025/ase-2025-papers/88/Faster-Runtime-Verification-during-Testing-via-Feedback-Guided-Selective-Monitoring
- MR-Scout https://doi.org/10.1145/3656340
- MR-Coupler https://conf.researchr.org/details/fse-2026/fse-2026-research-papers/200/MR-Coupler-Automated-Metamorphic-Test-Generation-via-Functional-Coupling-Analysis
- CASCADE https://conf.researchr.org/details/fse-2026/fse-2026-research-papers/131/CASCADE-Detecting-Inconsistencies-between-Code-and-Documentation-with-Automatic-Test
- DepState https://conf.researchr.org/details/issta-2025/issta-2025-papers/87/DepState-Detecting-Synchronization-Failure-Bugs-in-Distributed-Database-Management-S
- Onweer https://doi.org/10.1109/ICSME64153.2025.00031
- agent bugs https://conf.researchr.org/details/ase-2025/ase-2025-industry-showcase/23/A-Characterization-Study-of-Bugs-in-LLM-Agent-Workflow-Orchestration-Frameworks

TSE already establishes dynamic control-dependency-based test selection (DynaMOSA). TOSEM establishes transferable oracle synthesis (MR-Scout). Thus the paper cannot be justified simply by renaming target selection, temporal checking, or test oracles; the joint source-identity-bound fault-window compiler must exhibit its own distinct measured capability.

## 2. Anchor one REAL phenomenon rather than an acronym

**A runtime operation creates a consequential completion obligation; an interrupt, retry, cancellation or guardrail may occur before its matching discharge.**

The concrete observable manifestations vary:
- OpenAI Agents #3280 (old known source): output guardrail exception incorrectly accepted as successful final output.
- OpenAI Agents #4125 (old known source): approved, executed local tool has persisted call but lacks corresponding persisted result under streamed guardrail rejection.
- LangGraph #8393 (old known source): pending logical child ID readmitted on a parent retry. Community fix is NOT merged upstream.
- AutoGen #7956 (pre-registered DEVELOPMENT source): a cancelled async tool leaves its consumer waiting for a sentinel. Local causal source intervention makes cancellation terminate.
- Google ADK #7434 (pre-registered DEVELOPMENT source): repeated confirmation may consume the same approval again. Its author PR #7435 is closed UNMERGED; not a merged upstream fix. We have inspected this dev issue only, NOT run a native ADK experiment here.
- PydanticAI #9734: resource cleanup ownership disputed; correct scientific label UNKNOWN, not a confirmed bug.

The research target is not the identity of hidden states (OACR), not repair of already failed tool commands (BRFP), and not generic agents success rates. It is the *source-linked unfulfilled continuation/effect obligation* that determines a useful fault injection site and independent post-fault test oracle.

## 3. Precise mathematical hypothesis

Let P denote real SDK source; C independently sanctioned behavioral contracts; A permitted fault/test authority. The source analyzer constructs a conservative graph of activated actions, interruptible sites, and matched completion effects.

For obligation o, write

\[
o=(\kappa, \iota, a_o, D_o, E_o),
\]

where type \(\kappa\) could be stream terminal, accepted output, pending task identity, approval consumption or durable result; logical key \(\iota\) identifies the *same* invocation, queue or approval; \(a_o\) activates it; \(D_o\) are admissible discharges; \(E_o\) are observable evidence sources. Importantly, *the normative contract is NOT inferred solely from the potentially buggy program*.

The outstanding set after a trace prefix \(\pi\) is

\[
\mathcal D_C(\pi)=
\left\{o:\operatorname{activate}(o)\in\pi
\land \not\exists d\in D_o\
\operatorname{matches}_{\iota}(d,o)\land d\in\pi\right\}.
\]

This outstanding-obligation definition is **standard temporal protocol/typestate machinery**, not itself a new theorem.

Our *candidate technical contribution* is compiling and executing an **obligation interruption window**

\[
w=(o, a_o, f, d, \rho, \nu, \mathcal O),
\]

where \(f\) is a permissible fault cut between the activation and a required discharge \(d\); \(\rho\) is an independently checkable queue/approval/effect-ID binding proof; \(\nu\) is a normal negative-control run under matched authority; \(\mathcal O\) is an external native consequence observation and its legal contract.

A window produced by static source inspection does NOT imply an actual bug. It becomes an evidence-bearing test only after both control and native fault execution complete. The outcome is three-valued:

\[
\operatorname{Judge}(P,w,C,A)\in\{\mathrm{FAIL},\mathrm{PASS},\mathrm{UNKNOWN}\}.
\]

For example, a stream risk candidate requires source-observed task spawn of producer with the **same queue** as an awaiting consumer, a cancellable await before terminal signal, and no guaranteed finally-path discharge. This overapproximates legal executions; actual runtime reachability and timeout must be established separately. A lost remote write acknowledgment with no idempotency/reconciliation authority yields UNKNOWN; never assert exactly-once.

An ordinary budgeted coverage objective

\[
\max_{W\subseteq\mathcal W}\sum_{o\in C}
\mathbf1[\exists w\in W:\ w\text{ checks }o]
\quad\text{subject to }\sum_{w\in W}c(w)\le B
\]

is **classical maximum coverage**, not a novelty claim. A future original algorithm would need a demonstrably different scheduling objective or new ability, such as lowering invalid injections without losing real-fault detection.

## 4. Testable proposed algorithm

1. Extract activation and discharge candidates, their logical identity bindings, and possible interruption cuts from native source, not issue names.
2. Reject windows with disjoint resources or explicit guaranteed discharge on cancellation/fault paths. Store source SHA and relevant line/call/alias witness.
3. Map the remaining source-derived windows to **separately authorized external obligations**; unknown contracts never become automatic bugs.
4. Produce an admissible pair: ordinary completion plus fault while the relevant operation is in flight; use local fake model/tool, never touch production.
5. Execute on pinned original SDK with independent runtime evidence of end states, persisted results, approval consumption, or logical task admission. Missing control, ambiguous external effect, non-running fixture -> UNKNOWN.
6. Optionally test a narrowly sourced local discharge-edge intervention to rule out alternate causes. This is corroborating evidence, not a novel counterfactual principle.
7. Choose candidate windows at fixed execution and human-setup cost; report misses and invalid/UNKNOWN attempts as denominators.

This is a *mechanism hypothesis with a limited implementation*. The shared compiler currently supports **stream-terminal obligation** only. The author-installed native adapter for AutoGen is still manual and was written AFTER seeing a development issue. Other obligation types exist in the older issue-specific scripts but have NOT been generalized into this compiler.

## 5. Implemented v0, verified limitations, honest observed evidence

- **obligation_window_kernel.py**: scans arbitrary Python source, independent of upstream repo path/issue ID, for nested async producer, actual create_task formal→actual argument identity, awaiting consumer queue, abortable producer await and a normal-only None terminal notification. Emits plan and source SHA. No static BUG label.
- **native_window_executor.py**: regenerates candidate from source, checks plan was not forged, and permits one trusted local AutoGen adapter to execute its proposed normal/cancel pair.
- **test_obligation_window_kernel.py**: a tiny adversarial synthetic corpus and unit checks for finally, mismatched queues, no consumer/task, keyword args, alternate producer/await name and source tampering. First-cut version admitted an await after an unconditional return; it was repaired by simple block-local control-transfer pruning, and queue alias rebinding after task spawn is now excluded. This is *not* full interprocedural reachability.
- CI: https://github.com/MKLEE222/NLP/actions/runs/38117548112 : 36 tests, native real pinned AutoGen source selected 1 window at await line 1200 and discharge line 1214, normal local replay passed; cancellation timed out at 1.5s. Separately created local causal patch discharging in finally produced 0 candidates. This is **one known upstream development issue**, not an unseen discovery.

The comparison against a token-cooccurrence source scanner in unit tests is intentionally **weak sanity checking**. It does not establish lift over FailMapper, mature stateful/property testing, fault injectors, concurrency schedule tools or modern oracle inference.

## 6. Actual top-venue-grade research gates

- RQ1: Can obligation-window compilation automatically yield more **legal, executable** failure scenarios per setup-hour and CPU-minute than serious random/AST/stateful test baselines under the same API rights?
- RQ2: Do logical identity binding, guaranteed discharge checks and independent native obligations reduce false alarms, instead of declaring every sync/stream difference a bug?
- RQ3: Can the compiler transfer without a fresh handwritten detector/adapter *per issue*, or do the engineering costs dominate?
- RQ4: Is there distinct impact beyond the author's own regression test and a strong published tool? Existing methods catching some failures is allowed; total duplication of all capabilities/risks would reject a standalone method claim.
- RQ5: Are failure-window counts, source-validity, UNKNOWN, rare real faults, human setup time and false alarms properly grouped by **independent issue/repository** rather than duplicated scheduling paths?

Fair baselines:
1. Original SDK regression tests and scripted replay utilities.
2. Naive token AST scanner and uniformly/randomly sampled **legal** injected await positions.
3. Python stateful/property-based and concurrency schedule testing with equivalent instrumentation.
4. FailMapper/FRIES/Valg/MR-Scout/MR-Coupler style methods as **conceptual near-neighbors**; run an official or transparently adapted implementation only if its language, I/O and license contract genuinely matches. The FailMapper public Java implementation uses an LLM API and JUnit; do not claim it was run here or ask user to pay.
5. Ablation: remove alias binding, finally handling, contract gate, source re-verification, normal control, and obligation-diverse selection.

FROZEN DATA: sixteen title-triaged public issues in four frameworks; four dev issues read; TWELVE content-sealed confirmation issues **not read or used**. No confirmation opened until method, eligible-source denominator, native adapter policy, comparator budgets, source hashes, and STOP/UNKNOWN conditions are fixed.

GO: an original reusable component with substantial measured gain (coverage, validity, low false alarms, authority/cost) on untouched real tasks; or an independent, useful empirical benchmark with new findings. NO-GO for the method only if effects reduce to manually hardcoded known repro scripts and no distinct capability remains. Select venue **after** empirical contribution exists.


## 8. Second cold-start static precision pass (same day)

The initial generic obligation-window compiler would still report a syntactically present await after a direct unconditional return/raise, and could join the producer's original queue to an outer consumer that had *rebound its queue name* after task spawn. These were scientific validity defects, not new bugs in third-party SDKs.

We repaired the compiler to (a) stop a block-local candidate path at direct return/raise/break/continue transfers; (b) reject producer→consumer name matches when the outer queue variable is reassigned between the source task spawn and the awaited read. Existing lexical finally, missing consumer and mismatched queue controls remain.

The adversarial micro-source corpus now includes these traps. The retained tiny corpus has **15 source snippets with manually declared structural eligibility**: 5 eligible and 10 ineligible. The token-presence baseline intentionally does not check reachability/alias and is only a *sanity* comparator, not FRIES, FailMapper or a realistic property-based test method. These are **source-test-plan eligibility labels**, not 15 real runtime bugs or automatic held-out detections.

This narrow repair does not solve branching reachability, exception suppression, awaitable completion outside queue identities, false signals, or real remote effect observability. Such unsupported patterns must remain UNKNOWN/NO-CANDIDATE rather than being promoted to a correctness theorem.

Latest CI with both native execution and all regression tests:
https://github.com/MKLEE222/NLP/actions/runs/38117748435

Frozen 12 issue-confirmation records remain unopened. A possible next method improvement is typed activation/discharge across approval use and durable effect receipts, *but only if the same generic analyzer compiles them without a special bug-ID branch, with matched negative controls*.
