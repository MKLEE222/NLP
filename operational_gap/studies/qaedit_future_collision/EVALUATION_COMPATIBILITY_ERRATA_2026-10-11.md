# Evaluation compatibility errata — 2026-10-11

Written before any registered sequential development/confirmation model outcomes.

## The discovered mismatch

Our first Qwen CPU engineering pilot and the 8-row single-edit screen used a strict case/whitespace-normalized exact match and generation up to 48 tokens or EOS. These results are **not** WILD benchmark results. The published reference code at https://github.com/WanliYoung/Revisit-Editing-Evaluation/blob/61ccca3c1ecdb3976c9c0924e9352f3e64cc7d48/easyeditor/evaluate/evaluate_utils.py defines normalization that removes articles, punctuation and case differences, and generation that stops at period, newline or EOS (maximum 50 tokens). It uses normalized EM when no paid LLM judge is provided.

The authors used question-only input by default. Our Qwen2.5-Instruct instrument retains a fixed chat-formatted question instead, and is therefore **WILD-aligned**, not an exact reproduction.

## Registered measurement separation

1. LEGAL PRE-EDIT TRANSCRIPT: T_Q(s) is the text generated for allowed read-only questions. Q0, Q4, Q8 are fixed budgets. A collision requires literal equality after case-and-whitespace canonicalization. It does **not** use WILD answer normalization (which could inflate collision matches).
2. ANSWER SUCCESS: E(q,s)=1 if the freely generated natural-stopped answer equals the reference under the published WILD normalization. We report earlier strict results separately.
3. FUTURE-EDIT CONSEQUENCE: H(s,a) has three components: post-minus-pre target answer success; post-minus-pre paraphrase answer success; and locality perturbation fraction. The last is one minus the fraction of unchanged free-generated locality transcripts, over the 5 fixed locality questions. Locality gold correctness may be shown separately but does not replace stability.
4. AUTHORITIES: the risk policy sees only Q0/Q4/Q8 legal read-only text transcripts; held-out rephrase, locality and post-edit results are evaluator-only. Gradient, logit, model clone/reset, counterfactual edit and optimizer-state access are separate instrumented-oracle capabilities and must not be compared at equal black-box budgets.
5. EVIDENCE: any matched pair of sampled states establishes only a witness lower bound on operational ambiguity, not an estimate of the full-state supremum. Repeated locality tests across allocated roles are common outcome instruments and require clustered uncertainty/sensitivity analysis.

## Frozen records and gates

- Immutable original QAEdit source: 19,249 rows, SHA256 fab6328110cbb2912eb71945e11b12ef0606fcf5ddbd6a3fafd2fee6b6fdc4cc.
- First development engineering pilot (one history edit per side): Q0 collision FALSE, strict correctness divergence ZERO, observed generations changed.
- First strict eight-row edit viability check: 0/8 strict correct after 8-step LoRA, 7/8 generations changed and training loss declined for all eight. Not WILD-aligned.
- A separate WILD-aligned viability rerun uses **the same first eight development IDs** and identical editor settings, with only externally defined evaluation semantics corrected. Do not overwrite any earlier report.

If the WILD-aligned screen still shows no genuine target corrections, BLOCK the current editor configuration before expensive registered sequential experiments. Do not silently change rank, learning rate, data IDs, prompt format or metric to rescue it. Any new editor is an independently justified new research arm, with its own preregistration. Untouched confirmation remains unopened.

Source comparison: Yang et al., ACL 2025, The Mirage of Model Editing: Revisiting Evaluation in the Wild, https://aclanthology.org/2025.acl-long.745/.
