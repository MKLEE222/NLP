# Frozen holdout intake — development-only first read

Scope: frozen register `unseen_issue_registry_v1.json`, commit 06b5c5d40baffab59cacb347777d46a54a9735af. **All 16 IDs/roles committed before any issue body was opened.** Only 4 development bodies have subsequently been inspected. The 12 sealed-confirmation bodies/comments/patches remain unopened. Issue titles had been viewed in the initial search, documented as title-level exposure.

| Development issue | Native code/failure | Preregistered handling |
|---|---|---|
| microsoft/autogen#7956 | In-flight tool cancellation can propagate out of a gathering child without sending terminal stream sentinel, deadlocking consumer. Public ref 027ecf0a379bcc1d09956d46d12d44a3ad9cee14. | **First new mechanism pilot**: source-to-terminal-signal AST detection + actual fully local agent execution. Negative control: same tool completes normally. Never claim an unknown discovered bug. |
| google/adk-python#7434 | Replayed tool approval may execute a previously confirmed tool again. Closed; reported on author SHA 666c26db and a proposed fix. | Next development mechanism: approval-use ID/replay; independent event count, no real transfer. Compare to standard idempotency-handling contract, not assumed exactly-once remote payment. |
| agno-agi/agno#9955 | Loop iteration human-review reject/resume: nonstream passes resume parameters, streaming path allegedly restarts at index 0. Open. | Development audit only until code can run with local deterministic Loop steps. No blind cross-mode equality claim. |
| pydantic/pydantic-ai#9734 | Streaming async-generator cleanup apparently deferred on graph context exit; explicitly described by reporter as an **open ownership contract question**. | **UNKNOWN / contract contested** until official ownership semantics support a definitive post-exit cleanup obligation. Positive/negative source execution may verify lifecycle mechanics but MUST NOT label this alone as a framework bug. |

Development source inspection itself is not evidence of novelty. New cancellation generator profile is being constructed **after** reading AutoGen #7956 and so is **not issue-blind generalization**. The scientific test later requires unseen confirmed issues, per-project negative denominators, and cost-matched baselines.

**Current scientific target**: automate fault *boundary* identification, reachability-constrained control/fault pairs, and contract-appropriate terminal/lifecycle consequences beyond static exception search and authored reproductions. A strong existing test system is a comparator, not automatically a veto.

Rights: no hosted model, payment, production write or persistent side effects. All virtual SDK tests use local replay and process-local events. Preserve all no-candidate, unsupported-adapter, incompatible-runtime and UNKNOWN outcomes in the denominator.
