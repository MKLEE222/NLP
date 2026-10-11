# Direct-neighbor red team: LDFI, Sieve, and agent-runtime obligation windows

Date: 2026-10-11. Confidence: high for the *overlap* in mechanism class, unknown for incremental gain. Status: independent contribution NOT ESTABLISHED. We do not merge or open the 12 sealed confirmation issues.

## 1. The more threatening primary neighbors

**LDFI (Lineage-driven Fault Injection)**, Peter Alvaro et al., ACM SIGMOD 2015; operational deployment at Netflix documented at ACM SoCC 2016, *Automating Failure Testing Research at Internet Scale*. Core insight: use causality/data lineage of successful execution to infer fault combinations which could invalidate success. It is explicitly dependency/provenance-directed fault exploration, not random injection. Original Molly implementation is public: https://github.com/palvaro/molly . Description: https://disorderlylabs.github.io/ ; industrial paper: https://par.nsf.gov/servlets/purl/10053503 . A Python derivative Chaoxis exists (https://github.com/KDahlgren/chaoxis). **We must not call identity-bound fault selection, causal windows, or provenance itself new**.

**Sieve**, Xudong Sun et al., USENIX OSDI 2022, *Automatic Reliability Testing for Cluster Management Controllers*. Identifies promising fault sites from normal instrumented execution, builds fault plans, tests perturbation against normal run with differential safety/liveness oracles, automatically reproduces issues. Reports **46 serious bugs in ten controllers, 35 confirmed, 22 fixed, 3.5% false-positive rate**, and principled pruning of redundant plans. Strong code/benchmark artifact: https://github.com/sieve-project/sieve ; refereed paper: https://www.usenix.org/conference/osdi22/presentation/sun . It is structurally CLOSEST to a claimed "source-guided fault-window generator + normal-control validation"; do not sideline it because it targets Kubernetes rather than Python Agents.

**Efficient Reproduction of Fault-Induced Failures in Distributed Systems with Feedback-Driven Fault Injection**, Pan et al., ACM SOSP 2024, https://doi.org/10.1145/3694715.3695979 . This also occupies learning which injection schedules reproduce existing failures and reducing exploration cost. Our budget selector is not new.

These are **refereed systems top venues**; they are not direct Agent SDK paper results, but they dominate general claims about automated causality/provenance-guided fault injection and liveness detection. A novel Agent-targeted component needs a concrete capability absent or significantly costlier under these methods.

## 2. Against each neighbor: what we presently have

| Proposed idea | LDFI / Sieve already cover | Current code proves | What still might be ours (hypothesis only) |
|---|---|---|---|
| Find an interruption just before a required effect | LDFI infers causal dependencies and attack sets; Sieve proposes fault points from dynamic normal traces | AST discovers two templates in known SDK issues | Authenticated *normative* obligation extraction keyed by approval/tool invocation/session/result identity |
| Normal execution as a valid negative control | Sieve directly uses unperturbed versus perturbed executions | AutoGen completion / cancellation pair; LangGraph distinct ID pair | Comparability under native SDK permission/authority contract with low false alarm on invalid pair |
| Check liveness/safety | Sieve explicitly checks both | Local queue termination and logical duplicate admission | Correct status for ambiguous external side effects, unacknowledged commits and policy-conditional completion |
| Use provenance | LDFI is lineage-driven, and Sieve emits reproducible fault plans | Source SHA, queue alias, AST sites | Semantically verified identity/capability witness rather than syntactic dataflow alone |
| Reduce wasted test plans | LDFI/ Sieve/feedback-guided approaches prune, and DynaMOSA covers target selection | Tiny greedy diversity, trivial token comparator | A directly measurable valid-native-scenario/engineering-hour gain under equal rights and budget |
| Cross-system generalization | Sieve on ten controllers; LDFI at Netflix across microservices | AutoGen and LangGraph, 2 known issue units, two human-built native adapters | Cross-framework Agent SDK obligation compiler without per-issue hand-written harnesses |

The overlap is **substantive**. Calling our v1 an original new method at this point would be premature.

## 3. Revised defensible research question (not a claimed solution)

Can a developer-usable test compiler for real Agent runtime SDKs, with *independent normative approval/identity/durable-effect semantics*, automatically generate source-grounded legal failure windows and executable obligation checks that add verifiable capability (or cheaper high-validity native cases) over (a) dynamic trace-based Sieve-style fault plans, (b) LDFI-style dependency-directed fault sets, (c) native author tests, (d) Python property-based/schedule testing?

**Important difference to establish, not assume:** Agent tool approval is a *capability* with potentially one-shot authority; an already executed tool call may be externally committed even if a result is lost; a stream has a caller-visible liveness duty; a retry should be keyed to a logical task ID, not a Python object. These are real contracts in underlying SDKs, but source-only metadata cannot infer legal rights by itself.

Outcome must be tri-valued. If contract authority or committed effects are hidden, return UNKNOWN rather than claim true exactly-once or label a bug.

## 4. Scientific test design from the stronger neighbor analysis

**Freeze before confirmation:** 
- Register supported obligations, SDK versions, source eligibility and all adapters at the issue-independent *framework level*. No per-issue special scripts for confirmation.
- Use native test scripts/official replay models to run the same underlying workflow and fault budget.
- Implement at least a transparently **Sieve-inspired** native trace-fault differential baseline (not call it original Sieve, which targets Kubernetes). Adapt LDFI lineage scheduling only where trace/lineage data are available; missing instrumentable sources are incomparable and recorded as such, not zero scored.
- Give comparator the same setup labor limit, program/source access, side-effect permission, runtime observables, retries/fault budget. Publish the cost of adapter construction.
- Score eligible candidate/invalid/UNKNOWN/FAIL outcomes independently; false alarm by maintainer-documented semantics; bug identity unit is upstream Issue, not number of generated fault sites.
- Add negative controls for legal streamed differences and ambiguous lost-ack write; test-only source patch alone is not independent oracle proof.
- Do not treat synthetic source-snippet count, known source/fix contrast, or multiple entrypoints of one upstream issue as independently discovered bugs.

**Novelty gate:** additional contractual test *capability* or a substantial reduction in invalid/false-positive native tests on unseen issue families, with ablations showing identity/rights/contract obligation compiler matters. Merely reproducing LDFI/Sieve fault-window targeting in Agent SDK vocabulary is NO-GO for a standalone algorithm but could still contribute a rigorously constructed benchmark or empirical measurement.

## 5. Current evidence recited without inflation

- 41 repo tests, 4 GitHub Actions jobs green on run https://github.com/MKLEE222/NLP/actions/runs/38117897011 .
- Two source obligation subtypes in a generic Python AST compiler: async stream completion and pending logical task admission.
- Two previously reported real upstream issues across AutoGen and LangGraph, three source versions; manual native adapters. AutoGen observed cancellation timeout; LangGraph _call observed a duplicate admission on old source and none on *unmerged* community patch. No new bug, no fair Sieve/LDFI comparison.
- Twelve preregistered title-exposed but **content-sealed** confirmation issues remain untouched.

## 6. Source citation discipline

Use the refereed SIGMOD/OSDI/SOSP/SoCC sources and their own public artifacts to make novelty comparisons. Do not cite LDFI or Sieve as though they prove our Agent-specific mechanism impossible; do not cite an engineering GitHub README as a peer-reviewed accepted method. No automated official LDFI/Sieve run has been performed. The current GitHub prototype is a *hypothesis*, not a demonstrated advance over them.
