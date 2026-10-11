# Development comparison protocol: Obligation-Delta vs classical source/trace ideas

Date: 2026-10-11. Status: preregistered *development* comparison, not unknown-bug confirmation or validated original algorithm.

## Design question

With identical pinned Python source, allowed execution modes, normal-path observations, cut search universe and CPU/test budget, does a **contract-gated identity-aware obligation checker and cut selector** produce more valid, auditable native tests than generic source, lineage or trace heuristics? Separate structural candidate validity from real bug detection. **Never use known bug/fix labels when selecting cut sites.**

## Frozen methods and information

B0 (uniform): seeded deterministic permutation of independently enumerated candidate cut locations, seed OD-DEV-20261011-V1.

B1 (local-source): test candidate where same function contains an await before a queue sentinel, or a stored task ID comparison appears near task scheduling. Ignores cross-function queue alias and protected cleanup.

B2 (lineage-inspired): use local AST/dataflow to connect producer queue to consumer awaited read or pending task to candidate task ID. Does not reason whether finally or a contractual repair alternative guarantees discharge.

B3 (trace-inspired): use a *healthy execution trace* to restrict to observed cut sites, then choose downstream/near-output cuts; setup tracing expense counts. Without a real trace the method status is UNSUPPORTED, not FAIL/PASS. These B2/B3 are transparent **local adaptations**, NOT implementations of SIGMOD LDFI/Molly and OSDI Sieve. Sieve itself depends on instrumenting the Kubernetes controller environment; Molly reasons over distributed causal lineages with SAT.

M0 (source-window compiler, existing): typed source candidate with alias and cleanup checks; fixed greedy obligation diversity.

M1 (our proposed increment): a *contract-gated witness compiler* that takes exactly the source candidates and a separate, non-issue-specific normative contract (allowed fault, authority, required evidence, logical identity). Generates a normal/fault pair, requires independent native observations and correct identity-specific discharge, and yields PASS/FAIL/UNKNOWN. An UNKNOWN is not a PASS and isn't a silent filter. The M1 selector cannot use source issue IDs or expected outcomes.

Ablations: no alias binding; no guaranteed-finally filter; no normal control; no contract gate; no effect-ID matching. Report M0-vs-M1 ties honestly.

## Denominators and rights

D-A (development source corpus): the pre-existing 15 small async snippets, new small negative examples, two task-ID snippets, plus same real pinned source files as the existing development runs. Source snippet ground truth is *static eligibility* only; it was developed alongside the methods and cannot establish independent precision/generalization. Syntax controls, false positives and no-candidate belong in denominator.

D-B (native historical known issues): AutoGen #7956 and LangGraph #8393, optionally OpenAI Agents #3280/#4125 if a reusable normalizer is implemented. Count independent issue units and SDKs, not code sites, variants or repeated jobs. These bugs were public before method creation. Reuse the existing source SHA and native receipts; add native runs only as necessary to compare permitted executions.

D-C: the **12 precommitted content-sealed confirmation issues** remain unopened and unsearched. Their titles were seen in the initial search; record this nonzero exposure. Do NOT inspect any issue bodies, comments, fix diffs or outcomes to design new patterns. No confirmation before a frozen general adapter policy, strong baseline budget, legal contracts and predeclared UNKNOWN treatment.

Fairness: all policies see exactly the same source and same optional normal trace, source access and permissions. Budget B in {1,2,4} counts candidate execution attempts; source parsing plus tracing time must be measured. Source-only method requires an executable trusted SDK adapter before claiming native success. If baseline suggests a site unsupported by an adapter, record UNSUPPORTED, not zero technical efficacy. Record author-side integration effort honestly (possibly UNKNOWN).

Outputs: candidate count, valid candidate subset, eligible/unsupported, native execution attempts, valid control/fault pairs, PASS/FAIL/UNKNOWN, false alerts on normal/legitimate variants, CPU runtime and setup/instrumentation costs. Never call known historical failure a new bug. Structural metrics and native metrics must be in separate tables.

## Strong published neighbors: explicit non-claims

- LDFI, SIGMOD 2015: https://doi.org/10.1145/2723372.2723711 and https://github.com/palvaro/molly .
- Sieve, OSDI 2022: https://www.usenix.org/conference/osdi22/presentation/sun and https://github.com/sieve-project/sieve .
- FailMapper, ASE 2025: https://conf.researchr.org/details/ase-2025/ase-2025-papers/212/FailMapper-Automated-Generation-of-Unit-Tests-Guided-by-Failure-Scenarios .

Published tools are NOT claimed to have run unless their original source was executed. Similarity of general fault planning doesn't forbid an Agent-specific mechanism, but a label or small hand-authored identity predicate is NOT enough to claim a research advance.

## Stop/pivot

If M1 ties M0 in known dev cases, report TIE; prioritize additional non-bug-specific obligation generalization and a supported serious comparator before touching confirmation. If structural micro-corpus advantage alone exists, call it engineering sanity only. No journal, production write, paid LLM API, or merging PRs.
