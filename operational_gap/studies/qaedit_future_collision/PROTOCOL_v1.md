# QAEdit Future-Consequence Collision — preregistration v1

Date: 2026-10-11. State: protocol selected; real model experiments NOT RUN.
This document is public protocol only. Do not upload the private manuscript or frozen original carrier results into this repository.

## One falsifiable phenomenon

After distinct *real* sequential knowledge-edit histories, two writable model states can give the **same legal pre-edit QA answers**, yet respond materially differently to the **same subsequent edit**. We call an observed instance a *history-conditioned future-consequence collision*; this is a descriptive diagnostic, not an asserted new algorithm or mathematical theorem.

The distinction to test is pre-edit observational equivalence versus post-edit **incremental** outcome, not merely ordinary forgetting or failure under teacher forcing. The published ACL 2025 QAEdit/WILD paper already establishes substantial evaluation and sequential-edit degradation; ACL 2025 NSE, EMNLP 2025 MicroEdit, and ICML 2026 StableEdit address editing dynamics. None should be renamed as our contribution. Our unresolved question is whether pre-edit observations at realistic cost can **forecast or exclude** unequal next-edit consequences beyond strong baselines.

## Grounded, separately sourced materials

- Data: WenDingY/QAEdit, 19,249 public QAEdit records, fields Source, prompt, target, subject, rephrase, locality; dataset https://huggingface.co/datasets/WenDingY/QAEdit (dataset license listed as Apache-2.0). Use ONLY legal public dataset access; store SHA256 of downloaded bytes.
- Reference evaluation: WanliYoung/Revisit-Editing-Evaluation WILD *free generation*, not ground-truth teacher forcing: https://github.com/WanliYoung/Revisit-Editing-Evaluation
- First carrier: Qwen/Qwen2.5-0.5B-Instruct (public Apache-2.0 model): https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct . Pin the resolved immutable checkpoint SHA before any model run.
- First persistent editor: a **single** LoRA adapter repeatedly updated by one fixed algorithm, not a freshly reset adapter at every turn; rank 8, alpha 16, query/value projection targets, AdamW learning rate 2e-4, 8 steps per edit, max training sequence 320 tokens, max generated answer 48 tokens, greedy decoding, seed 20261011. These are *proposed fixed knobs*, not confirmed supported hyperparameters; freeze an executable configuration/eligibility audit before running. Do not silently change them to obtain a stronger effect.

## What is legal before the next edit

Primary *black-box read-only* contract: a decision policy knows the next edit request (question + intended answer), can ask the current model that question and, at additional budget levels, ask a fixed set of eight public sentinel questions. It receives only generated text under fixed greedy decoding; no current parameters, gradients, private edit history, true future edit result, weight reset, oracle Jacobian, future rephrase, or locality outcome. The evaluator can clone a checkpoint to apply the same next edit to paired states, but the policy cannot perform this free.

Contract Q0: one prospective question. Contract Q4: the prospective question plus the first 4 fixed sentinels. Contract Q8: the prospective question plus all 8 sentinels. A collision means all normalized generated strings match **exactly** across both states at a given budget. Do not relax this matching rule after observing outcomes. Include forward-call and generated-token costs. Report the exact collision denominator for each budget.

Separate *instrumented oracle* contract: parameter access, gradients, exact clone/reset, probe writes and singular-value/Taylor diagnostics. These may explain the phenomenon but must not be compared as if they consumed the black-box budget. Direct finite-write-and-restore simulation is also an oracle unless that ability is actually permitted.

**Mathematical correction to v2:** use a general observation map T_Q(s) = Obs(Q,s). The earlier write-difference transcript R_p(u;s) is only one permitted observation operator **when** reversible write probes exist; setting u=0 in a write-difference map supplies no observation. For read-only QA, T_Q returns free-generated answers (a discrete transcript, no differentiability or Euclidean Jacobian assumed). The indistinguishability relation is exact transcript equality, so Gamma_0(Q) remains meaningful with a discrete equality pseudometric. Do not report numerical Jacobian rank for these text transcripts.

## Independently measured future consequence

At a history checkpoint, take two independently accumulated adapter states. For each **identical** held-out future QAEdit action a, clone each state, execute the fixed next LoRA update, evaluate without teacher forcing. For state s:

H(s,a) = (delta target exact match, delta held-out rephrase exact match, delta mean locality accuracy), where delta is AFTER minus BEFORE this next edit on the same state. Also preserve the separate before/after values and generated text. Primary pair divergence is the maximum componentwise absolute difference ||H(s,a)-H(s',a)||_infty.

A witnessed lower bound is the maximum divergent H among **precommitted** paired states/events whose T_Q transcripts match. This is only a lower bound on the full-state supremum Gamma_0; finite sampled states do not estimate the universal supremum. A target/rephrase binary disagreement gives a unit divergence; alternatively a locality mean divergence of at least 0.20 is materially notable. Report all raw values and all matched/candidate denominators.

## Source splitting and strict independence

Use the scripts/prepare_manifest.py deterministic subject/QA-paraphrase connected-group split and stable SHA256 IDs. Never random-row-split subjects. Partition groups into hist_dev (46%), next_dev (20%), sentinel_dev (6%), hist_confirm (14%), next_confirm (10%), sentinel_confirm (4%) using seed "qaedit-future-collision-v1". Data must have required fields. File hash, group sizes, invalid rows and any content collisions are audited before model execution.

The development sample uses 4 *pairs* of disjoint edit histories (16 edits/history), checkpoints {8,16}, 4 held-out future actions per pair/checkpoint. The untouched confirmation sample uses 16 pairs (32 edits/history), checkpoints {8,16,32}, 4 independently held-out future actions per pair/checkpoint. Eight sentinel questions are selected deterministically per phase. No future edit from history pool; no sentinel from next-action pool. No state or future action can be selected after inspecting prospective consequences. A manifest may be public (IDs only), but confirmation predictions, outcomes and analysis must remain unopened until the freeze.

If any history edit violates the preregistered training contract (sequence too long, numerical error, impossible update), record the exact block and its denominator; do not repair a history or replace it with an easier example. Before eligibility filtering, specify a fixed baseline-wrong criterion if restricting to correction-needed edits, and preserve exclusion counts by source.

## Baselines and alternative explanations

At equal observation/call/token budgets: (1) no preflight, (2) prospective question only, (3) 4/8 random legal sentinel questions, (4) direct answer/log-prob prediction (log-prob access is its OWN priced contract), (5) nearest-neighbor predictors fitted only on development trajectories, (6) a simple empirical uncertainty or disagreement score. Instrumented gradient/low-rank/Taylor methods and next-write-and-restore are explicitly **separate higher-privilege ceilings**.

Alternative explanations that must be excluded: trivial baseline quality differences, an update operator that rarely changes any answer, malformed/ambiguous target answers, duplicate subject leakage, source distribution imbalance, low-capacity model saturation, ordinary cumulative forgetting, teacher-forced answer leakage, and larger probe/result budgets masquerading as one query.

## Metrics and decision gates

- Input validity: file hashes; usable records by source; group split leakage = 0; no future/rephrase/locality field used by the policy beyond the stated authority.
- Editor viability: report actual edits attempted, numerical failures, target improvement under greedy WILD evaluation, cumulative degradation, and whether a persistent adapter state genuinely changes. If almost no actual edit changes occur, stop; do not change the editor after inspecting confirmation.
- **Primary observation:** for Q0/Q4/Q8, collision rate (# matching pre-edit transcripts / # precommitted candidate state pairs) and discrepancy rate (# materially divergent H / # matching pairs). Report both to avoid selection-by-collision distortion.
- **Prediction:** false-safe rate, difference in bounded consequence risk, action ranking error, abstention/coverage, and total observed computational budget on independent history pairs; report no-probe and direct behavioral baselines.
- Bootstrap confidence intervals resample entire independent history pairs; shared future questions and checkpoints are not independent observations. Also report results per model/source and each checkpoint, without cherry-picking.
- **Gate A:** nonzero, numerically reproducible state changes and reasonable legitimate edits; otherwise STOP (editor/carrier failure).
- **Gate B:** at least 12 distinct independent confirmation pairs exhibit transcript match, AND among the precommitted matched cases at least one material future-consequence divergence, with quantitative uncertainty reported. If not, the primary operational ambiguity claim is unsupported; STOP without changing thresholds. No statistical novelty claim from one witness.
- **Gate C:** cost-matched consequence forecaster beats strong behavioral predictors on untouched confirmation with positive pair-clustered interval for the preregistered primary metric; otherwise no new predictive method claim. Only design a novel intervention after A/B provide evidence; not before.
- **Gate D:** rerun another editing mechanism or model (not just a seed), with the same information contract; if effect does not transfer, narrow the claim.

## Stop rules, provenance and permissions

No new priced external API. Do not run GPU/model experiments in GitHub Actions; CI executes **only the deterministic manifest and toy integrity tests**. Never claim that a passing CI job confirms the neural hypothesis.

No manuscript venue selected. Existing LoRA/GRACE/E2/KS frozen evidence remains unchanged. Open a new research line with a separate manifest and experiment ledger. Report null/blocked states and raw denominators. If the next-edit hypothesis fails, retain the corrected mathematical object as a scoped diagnostic rather than inventing a new operator.

## Publication nearest-neighbor audit

- Yang et al., ACL 2025, Mirage: https://aclanthology.org/2025.acl-long.745/
- Jiang et al., ACL 2025, NSE: https://aclanthology.org/2025.acl-long.815/
- Wang et al., EMNLP 2025, MicroEdit: https://aclanthology.org/2025.emnlp-main.1719/
- Ma et al., ICML 2026, More Edits, More Stable: https://proceedings.mlr.press/v306/ma26f.html
- Huang et al., NeurIPS 2024 / GRACE and classical observability / two-point minimax: mathematical ancestry, NOT proposed new proofs.
