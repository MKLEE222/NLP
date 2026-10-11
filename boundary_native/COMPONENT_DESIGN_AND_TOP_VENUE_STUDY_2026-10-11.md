# Native Agent Runtime Failure Boundaries: Accepted-Paper Methods and Candidate Components

Date: 2026-10-11. Status: SCIENTIFIC SCOUT, not a validated new algorithm or full benchmark. Public research-only tests; no private manuscripts or API secrets.

## A. Corrected principle: a strong baseline is not a kill switch

Existing techniques often cover the general *topic*, while accepted new methods contribute a new source-level component, coverage algorithm, trusted oracle, realistic environment, measurement contract, or cost frontier. A baseline that catches some or even all *currently inspected* bugs does not close every improvement opportunity. But any claim that our component has unique impact needs direct head-to-head comparison with that baseline on independent real tasks.

### Actual accepted precedents: mechanism, not buzzword

- NeurIPS 2024 SWE-agent: an agent-computer *interface* changed the agent's ability to inspect/edit/test repos. Its novelty was a supported interface mechanism with measured effects, not the fact that agents existed. https://proceedings.neurips.cc/paper_files/paper/2024/hash/5a7c947568c1b1328ccc5230172e1e7c-Abstract-Conference.html
- ICLR 2024 ToolEmu: tool-execution emulation plus scalable safety evaluation, validated for realism. https://proceedings.iclr.cc/paper_files/paper/2024/hash/7274ed909a312d4d869cc328ad1c5f04-Abstract-Conference.html
- ISSTA 2024 distinguished paper Mulpor: statement, expression and token level patch generators with a selection heuristic. A new composition and selection mechanism can be impactful despite competent predecessor methods. https://2024.issta.org/details/issta-2024-papers/124/One-Size-Does-Not-Fit-All-Multi-granularity-Patch-Generation-for-Better-Automated-Pr
- ISSTA 2024 MicroRes: failure injection, performance-to-user degradation contributions and resilience index. New concrete measurement mechanisms can count. https://2024.issta.org/details/issta-2024-papers/27/MicroRes-Versatile-Resilience-Profiling-in-Microservices-via-Degradation-Disseminati
- ICLR 2025 tau-bench: realistic stateful policy tools, real database end-state evaluation and repeated-trial reliability. https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b126cc38b8638e07bef37e7b2bb72bf-Abstract-Conference.html
- NeurIPS 2025 main-track AgentIssue-Bench: 50 reproducible real agent-system issues with executable environments and failure-triggering tests. A benchmark paper can be substantial without a new repair operator. https://proceedings.neurips.cc/paper_files/paper/2025/hash/45f0ab2d264a057113492b35c5ef47f2-Abstract-Conference.html
- ACL 2026 main-track SGVEF-LOOP: feedback, progressive topological exploration and fact-grounded metamorphic evaluation for MCP agents. Being another metamorphic technique did not preclude a new guided mechanism and meaningful empirical contribution. https://aclanthology.org/2026.acl-long.1224/
- Closest stress test: ISSTA/ECOOP 2024 DSTest intercepts messages to systematically explore concurrency without changing application source. Do NOT claim fault schedule exploration itself is original. https://2024.issta.org/details/issta-ecoop-2024-tool-demonstrations/11/Generalized-Concurrency-Testing-Tool-for-Distributed-Systems

We must distinguish ISSTA full technical papers from ISSTA tool demonstrations, and IEEE AITest papers from ICLR/ACL/NeurIPS main-track accepted papers.

## B. Native sources — mechanisms are heterogeneous

Source reports are leads unless independently reproduced on pinned native SDK commits. The first issue below has now been reproduced before/after by us using the public SDK FakeModel and no external model API.

1. OpenAI Agents SDK #3280, May 2026. Streamed output guardrail ordinary exception swallowed and answer accepted; fixed in PR #3281. Native reproduced by our own GitHub Actions run 38114666257. Pre-commit 683b6e79e5b4a2e3fc3cac80b556101ce778beae; merged fix 4a3d33e44fcd5051ff1b08f136ca61aef7824e63. Normal guardrail succeeds both versions. This run did NOT test the separately reported concurrent sibling-cleanup race.
2. OpenAI Agents SDK #4125, Aug 2026. On resumed human-approved tool, streamed output-guardrail trip could persist a function_call with no paired function_call_output while normal run preserved both. The stream path was *worse* and should not serve as the canonical reference. Fixed in PR #4126. Native reproduction pending.
3. OpenAI Agents SDK #4775, Aug 2026. Lost session-append acknowledgement after actual commit could duplicate pending input upon retry. Fixed/closed. Native reproduction pending.
4. LangGraph #8393, July 2026. Parent retry may schedule a duplicate child task because in-flight task object compared against an ID. Reporter reproducer available; issue open. Native reproduction pending.
5. LangGraph #6792, Feb 2026. Subgraph interrupt/resume lost output reuse, replayed prior step and duplicated interrupts. Reporter reproduction; issue open. Native reproduction pending.
6. CrewAI #7252, Sep 2026. Async guardrail rejection retry enters sync executor and treats coroutine as dictionary. Open reporter issue; native no-API reproduction pending.

The source-author-created issue labels and closure states do not themselves prove bugs remain on latest main.

## C. Candidate technical mechanisms to investigate, not claim yet

### Candidate A — Boundary-aware schedule/test synthesizer

Extract executable SDK entrypoint edges and source-allowed transitions (normal, async, streamed, approval resume, guardrail failure, timeout, cancellation, lost acknowledgement). Inject faults at reachable control/effect/await boundaries rather than uniformly random points. Construct paired controls with the same logical operation and observed policy.

Distinguishing hypothesis: a mechanism that targets guardrail/commit boundaries may find independently verified semantic regressions with fewer invalid injected schedules and fewer execution resources than general fault injection or random/metamorphic test generators.

Immediate threat: DSTest, MORPHAGENT, SGVEF-LOOP, bounded model checking, property-based stateful testing and classical concurrency tools.

### Candidate B — Effect-lineage obligation oracle

Represent each execution as a *partial order* of logical request IDs, approvals, guardrail verdicts, local and remote side effects (with confirmation state), persisted call/output pairs, cancellations and final accepted status. Verify contract-specific obligations rather than full byte-for-byte sync/stream equality.

For example: if output guardrail evaluation raises, a final answer must not be accepted. A successful approved tool call recorded in a durable session must have its corresponding result or an explicitly defined recoverable unknown status. Native contract permissions may differ for blocking versus parallel guardrails: do NOT assert equivalence not guaranteed by the SDK.

Never promise true exactly-once remote effects where a network acknowledgement may be lost and there is no cooperating idempotency key or reconciliation API. Failure can require a FAIL_CLOSED/UNKNOWN state rather than replay.

Distinct possible capability: construct and check the smallest auditable, authority-sensitive *obligation projection* of a multi-mode agent runtime, including state in session and external effects. Prior: linearizability, temporal properties, sagas/outboxes, event sourcing and API testing.

### Candidate C — Patch-sensitive evidence generator

Generate a witness triplet for each natural bug: original version failing its documented contract, independently reviewed fix passing, and matching negative control preserving legitimate operation. Optionally synthesize additional failure schedules around the fixed boundary, without violating SDK-documented mode differences.

Distinct possible capability: automatically detect an overfitted fix that passes the issue reproducer but violates adjacent failure path contracts. Prior: mutation-based test augmentation, differential testing, patch validation, FixCheck/STING, existing CI.

These candidates may be combined **only after evidence shows their separate contributions**. A component is not novel simply because it has an independent name.

## D. Trial success criteria (per mechanism, not blanket NO-GO)

- V0: real native source failure reproduced on pinned pre/post commits without paid APIs; compare allowed normal path to fault path.
- V1: two mechanism channels other than guardrail exception (e.g. approved effect-output orphan, lost-ack input duplicate) independently reproduced and attributable; across SDKs preferred.
- V2: smallest component supplies a demonstrably different output: valid inferred test pair, typed effect lineage, reduced false alarms or more fault coverage, compared to generic fixed assertions.
- V3: same-source versions and same budget comparisons against standard issue tests, random and boundary-guided injection, generic metamorphic transformations, and property-based or schedule tools. Report bug identities and true/false alarms plus execution costs.
- V4: ablation isolates the novel piece(s): reachability guidance, effect identity, approval/status obligations, extra test synthesis.
- V5: hold out whole upstream issues and frameworks; do not cherry-pick bug reports after results.

**Allowed outcomes:** (i) original agent-runtime testing component with credible measured gain, (ii) independently useful real-defect corpus and empirical finding, (iii) scoped negative result, (iv) another technical direction if mechanism saturates. Matching a strong comparator on one positive control is NOT automatic program death.

## E. Practical rights and difference from existing projects

This is testing of native Agent SDK lifecycle and repair verification. BRFP focuses on authorized local action repair after tool failure; OACR studies representational operational adequacy. Do not conflate these with a software test generator for async guardrails and effect persistence.

No secrets, new paid API, production writes, private files or hosted LLMs. Do not select a journal before the mechanism and native effect evidence exist.
