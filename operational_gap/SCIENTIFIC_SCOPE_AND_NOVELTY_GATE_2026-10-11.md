# Scientific scope and novelty gate — 2026-10-11

**Status:** research-direction correction; NOT an empirical result, not a verified algorithm and not a venue selection. This document **does not change** the frozen QAEdit v1 sample IDs, role allocation, Q0/Q4/Q8 contracts, confirmation set, Gate A failure or archived tests. It is a **new candidate scientific framing**, and any new arm must be preregistered independently.

## 1. What we almost did wrong

The existence of two model states with identical pre-edit QA answers and different responses to the same next edit is NOT, by itself, a novel scientific theorem. Different hidden states that agree on limited observations need not be equivalent for future actions. This is standard partial observability; predictive state representations (PSRs), observable operator models (OOMs), and bisimulation/behavioral metrics explicitly study action-conditioned predictions and state equivalence.

2026 literature makes the application novelty risk acute. Gu et al., ACL 2026, "Diagnosing Hidden Instabilities in Model Editing via Uncertainty Quantification" shows that conventional editing metrics miss accumulated interference, and proposes perturbation/UQ detection. Yang et al., ACL 2025, "The Mirage of Model Editing" already documents failure in realistic sequential editing and introduces QAEdit/WILD. Ma et al., ICML 2026, "More Edits, More Stable" shows early updates can *improve* later updates and provides lifelong-normalization theory/StableEdit. NSE (ACL 2025), WISE (NeurIPS 2024), and other sequential editors already seek to improve editing reliability.

Therefore do NOT claim as novel: (a) limited reads cannot identify all hidden states; (b) repeated edits can interfere; (c) current single-edit scores hide instability; (d) uncertainty can reveal hidden conflicts; (e) later updates depend on edit history; (f) an ordinary two-point minimax lower bound.

## 2. Scientific object, without tying it to QAEdit or NLP

A stateful adaptive system has hidden persistent state s, a legal observation contract I, a future authorized update/action a, a state transition U(s,a), and a consequential evaluation function g. Only the evaluator, never the decision policy, may clone or intervene freely to measure the ground truth:

- Observed pre-action transcript: Z_Q(s) = Obs_I(s; Q), where Q is a legal read-only or otherwise authorized probe protocol with cost c(Q) <= B.
- Prospective effect: Y_a(s) = g(U(s,a)) - g(s), evaluated under a fixed outcome task, units, and horizon.
- Decision: d = pi(Z_Q(s), a, optional *permitted* history), such as execute / defer / route to costly review / select update operator. An observation contract must say whether edit history, logits, gradients or adapter weights are allowed; do not silently assume them.
- Standard Bayes-risk comparator (not proposed as a new theorem): R_B = inf_{Q:c(Q)<=B, pi} E[L(pi(Z_Q(s),a),Y_a(s))], for an explicit observational population and downstream loss. Unobserved states and counterfactuals are evaluator-side labels only.
- Diagnostic witness (already preregistered for QAEdit): equal finite pre-action transcripts while next-action effects differ. Exact transcript collisions are ONE sufficient witness, not a required definition of the broader prediction problem.

This definition intentionally applies to **model editing** and potentially **online/continual adaptation**; a one-model, one-benchmark result does NOT license a cross-domain theorem.

## 3. Candidate phenomenon and the actual research fork

A worthwhile empirical finding would be an **action-conditioned pre-intervention decision blind spot**: conventional legal behavior/uncertainty/history summaries confidently identify two states or interventions as similarly safe, yet held-out future-action loss differs materially. This is only a *diagnostic gap* when it survives strong, cost- and information-matched baselines.

A worthwhile **method claim** (still UNPROVEN) additionally requires a pre-action, non-destructive, legally deployable protocol to reduce *held-out decision regret or false-safe rate* at the same total forward call, scalar output, token, wall-time and privileged access budget. A sophisticated new operator is not justified until this empirical increment is observed. If all existing baselines resolve the risk gap, this is a valid negative boundary, NOT an invitation to rename the baseline.

These are hypotheses subject to falsification. They are NOT theorems or observed new model-editing effects.

## 4. Three layers and independent stop rules

| Level | Question | Required counterfactual/comparator | Failure means |
|---|---|---|---|
| A. Scientific necessity | Is there a measurable and consequential pre-action gap on real, viable updating systems? | Matched held-out action, ordinary forgetting and base quality controls | No new mechanism; retain a bounded negative result |
| B. Relative originality | Is this more than PSR/partial observability, behavioral equivalence or 2026 UQ post-edit instability? | PSR-style observable history predictor; uncertainty and semantic perturbation; direct behavioral preflight; matched edit-history features | Do not call discovery novel |
| C. Actionable gain | Does a *permitted* pre-action policy add information and lower future decision loss under realistic budgets? | No-probe, Q0, random Q4/Q8, calibrated confidence/UQ, nearest-neighbor behavior, history-based prediction, extra-privilege oracle separately | No novel method claim; report empirical frontier only |

A counterexample that merely uses more observations, labels, answer normalization, privileged gradients, cloned future edits or uncharged resets FAILS B/C even if numerically impressive.

## 5. Research scope: enough breadth to test transfer, not uncontrolled sprawl

**Carrier E — knowledge editing (currently blocked, not abandoned):**
- QAEdit/WILD is a concrete controlled stress test, not the definition of the phenomenon.
- The precommitted Qwen2.5-0.5B 8-step LoRA engineering carrier produced 0/8 WILD-aligned corrected answers. Gate A = BLOCK; do not open the untouched 16-pair confirmation. These fixed results and evaluation errata remain immutable.
- Any replacement must be an **externally justified published editor on an actually supported model/compute stack**, with its own ARM_ID, source commit, model revision, evaluation compatibility test and separated eligibility development data. Prefer the original QAEdit paper's published FT-M setup for feasibility comparison (paper reports nonzero WILD reliability); WISE/StableEdit/NSE are SECONDARY capability candidates, not assumed drop-in CPU alternatives. No model run authorized by this note.

**Carrier C — independent non-NLP continual learning (candidate, not yet scheduled):**
- CORe50 (CoRL 2017, real continuous object recognition) or CLEAR (NeurIPS Datasets & Benchmarks 2021, naturally evolving imagery) offers a *different* state-update process and evaluation channel.
- Before activation, verify real access/data license, compute requirements, a published and working incremental-update baseline, legal non-destructive pre-update predictions, held-out future batches, independent histories and actual prediction utility. Do not force QAEdit exact-output collision thresholds or copy LoRA hyperparameters to this domain.
- Failure to transfer means any finding must remain knowledge-editing-specific. A second carrier is for generalization checking, not to manufacture the same outcome.

**Do not** add a third benchmark, new neural architecture or invented algorithm while Levels A/B are unsettled.

## 6. Metrics and fairness

- Decision-oriented primary scientific quantity: out-of-sample **future-action decision regret / false-safe at fixed coverage and budget** (only after a deployment decision and utility loss have been specified), with independent-history clustered uncertainty.
- Ancillary analysis: calibration/coverage, candidate-pair transcript collision rate and material divergence (report both denominators), continuous response differences, update success, utility and locality.
- Avoid counting shared QAEdit locality questions as independent observations; original source audit documented 21,128 normalized locality questions spanning multiple allocation roles.
- Separate observational privileges: black-box outputs, history-aware observation, probabilistic/logit access, gradients/weights, destructive replay with perfect reset. Do not compare costs as identical across tiers.
- Since an exact match after many text queries may be vanishingly rare, report its **precommitted sample denominator** and the whole risk-prediction population. It is impermissible to relax an existing collision criterion retrospectively; a probabilistic/approximate criterion requires a new, untouched arm.

## 7. Literature-level veto table (read before any new method)

| Prior result | Known content | What cannot be claimed | Concrete distinguishing test, if a gap remains |
|---|---|---|---|
| Singh, James & Rudary (UAI 2004), PSR | Observable predictions of action/observation tests summarize controlled dynamics | "Future effects need richer observable state" | Show gain beyond feasible PSR-style history/test predictors at equal allowed observations |
| Ferns, Panangaden & Precup (UAI 2004; SIAM J Comput 2011), bisimulation metrics | Action-conditioned state similarity and its consequences | "Observed equality is insufficient for action equivalence" | Define actual authorized observation/decision constraint plus empirical nontrivial gain |
| Yang et al. (ACL 2025), QAEdit/WILD | Sequential-edit and realistic-evaluation failures | "Single-edit benchmark conceals real degradation" | Predict a **specific not-yet-executed** update's risk, beyond direct pre-edit behavior |
| Gu et al. (ACL 2026), uncertainty diagnosis | Hidden interference; semantic-perturbation UQ diagnoses edit-induced instability | "Uncertainty reveals hidden editing instability" | Compare against pre-action UQ/perturbation at matched policy access; no post-edit lookahead |
| Ma et al. (ICML 2026), StableEdit | Lifelong normalization, history-dependent positive/stabilizing effects | "History influences edit outcome, possibly positively" | Treat its update strategy as a method and a baseline, not our theoretical discovery |
| WISE (NeurIPS 2024) and NSE (ACL 2025) | Existing lifelong and neuron-level edit solutions | "A distinct editing operator is automatically novel" | Only if a new preflight decision result remains after published editors are competent |
| Lomonaco & Maltoni (CoRL 2017) / Lin et al. (NeurIPS D&B 2021) | Real continual visual learning benchmarks | "Cross-task generalization" from one QA dataset | Genuine independent, task-preserving transfer required |

## 8. Governance and next allowed work

1. **No fresh confirm edits.** Keep current QAEdit Gate A failure and saved artifacts in place.
2. Prefer reference implementation feasibility audits (model, hparams, dependencies, GPU requirement, WILD evaluation) and data/license assessments over blind hyperparameter search. Author a new arm with explicit entry/stop rules before execution.
3. As a parallel non-experimental task, write the pre-action decision utility and the strongest feasible PSR/UQ/behavioral baselines. If the accepted prior work solves the same decision problem at the same costs, archive this direction as NEAR_NEIGHBOR_SATURATED instead of writing an invention.
4. Only after viable evidence for A and a real relative gap for B may algorithm design begin. The methods section may remain empty until then.

### Primary published sources (not evidence of this hypothesis)

- UAI 2004 PSR: https://web.eecs.umich.edu/~baveja/psrpubs.html
- UAI 2004 bisimulation metrics: https://mlanthology.org/uai/2004/ferns2004uai-metrics/
- ACL 2025 QAEdit/WILD: https://aclanthology.org/2025.acl-long.745/
- ACL 2026 uncertainty diagnosis: https://aclanthology.org/2026.acl-long.1502/
- ICML 2026 lifelong normalization/StableEdit: https://proceedings.mlr.press/v306/ma26f.html
- NeurIPS 2024 WISE: https://proceedings.neurips.cc/paper_files/paper/2024/hash/60960ad78868fce5c165295fbd895060-Abstract-Conference.html
- ACL 2025 NSE: https://aclanthology.org/2025.acl-long.815/
- CoRL 2017 CORe50: https://proceedings.mlr.press/v78/lomonaco17a.html
- NeurIPS Datasets & Benchmarks 2021 CLEAR: https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/2838023a778dfaecdc212708f721b788-Abstract-round2.html
