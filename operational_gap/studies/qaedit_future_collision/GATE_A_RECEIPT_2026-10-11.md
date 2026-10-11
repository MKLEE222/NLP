# Gate A receipt — 2026-10-11

**Decision: BLOCK_CURRENT_EDITOR_FOR_FULL_CONFIRMATION.**
This is an engineering/empirical eligibility failure for **one frozen editing recipe**,
not rejection of the operational-consequence question.

## Executed, immutable public source

- Dataset: QAEdit 19,249 rows from WanliYoung/Revisit-Editing-Evaluation,
  commit 61ccca3c1ecdb3976c9c0924e9352f3e64cc7d48,
  SHA256 fab6328110cbb2912eb71945e11b12ef0606fcf5ddbd6a3fafd2fee6b6fdc4cc.
- Core QA validity: 0 invalid / 0 duplicate; 17,509 connected subject groups;
  0 cross-role exact prompt/rephrase leakage.
- Locality caveat: 96,245 locality occurrences, 51,092 unique normalized questions,
  21,128 shared across assigned roles; 0 direct overlap with primary/rephrase questions
  and 0 normalized answer conflicts. Common test-instrument dependence must be modeled.
- Frozen manifest: 4 development history pairs / 32 prospective events;
  16 untouched confirmation pairs / 192 events. Confirmation has NOT been opened.
- Audit execution: https://github.com/MKLEE222/NLP/actions/runs/38110365128
  and https://github.com/MKLEE222/NLP/actions/runs/38110052554

## Carrier and exact editing recipe

- Qwen/Qwen2.5-0.5B-Instruct,
  checkpoint a338b55dd21219a5f4da42bc11a9313d1a27d4cc,
  float32 on CPU, no paid API.
- One LoRA adapter, q_proj/v_proj, r=8, alpha=16, 540,672 trainable parameters.
- AdamW, learning rate 2e-4, 8 gradient steps per edit, sequence max 320,
  fixed seed 20261011. Greedy free generation and no teacher forcing at test.
- The 8-row viability experiment is intentionally **single-edit with reset
  between fixed development IDs**, not a sequential-checkpoint estimate.
- No parameter search, replacement examples or reweighting after inspecting results.

## Results

| Arm | Sampled events | Generation changed | Correct answer after edit | Outcome |
|---|---:|---:|---:|---|
| CPU engineering dual-history pilot | one future action, two one-edit histories | yes | neither corrected strict target / rephrase / locality | Q0 collision FALSE; H divergence zero under strict metric |
| Strict single-edit viability | first 8 frozen dev history-A IDs | 7/8 | 0/8 strict EM | Not eligible as WILD reproduction |
| WILD-aligned single-edit viability | **the same 8 IDs** | 7/8 | 0/8 WILD-normalized EM; 1/8 answer string contained gold (not full correct) | Gate A BLOCK |

All eight single-edit training losses fell under both evaluation contracts. Therefore
training loss and visible text changes are not sufficient evidence of factual correction.
At most this is a small, TQA-heavy development screen; do not infer general model-editing
ineffectiveness from it.

Immutable runs:
- First CPU pilot: https://github.com/MKLEE222/NLP/actions/runs/38110139104
- Strict screen: https://github.com/MKLEE222/NLP/actions/runs/38110372259
- WILD-aligned screen: https://github.com/MKLEE222/NLP/actions/runs/38110498168

## Evaluation correction and claim limits

Published WILD (Yang et al., ACL 2025) uses natural stopping and normalized EM
for no-API grading. Our original pilot did not fully match and is preserved as a
historical engineering output; independent WILD-aligned rerun does not overwrite it.
The aligned Qwen run still uses a specified chat-formatted prompt, so it is not
an exact question-only replication of the ACL authors' implementation.

A future consequence should measure target/rephrase post-minus-pre normalized EM
and **locality pre/post answer stability**, not merely locality correctness.
Pre-edit transcript collision is separately assessed with case/whitespace canonical
text equality, never normalized-answer equality.

No registered 8/16 or 8/16/32 history checkpoint has been run; no new
consequence-ambiguity witness was found; no operational novelty claim is licensed.

## Stop and next legitimate research step

STOP this editor configuration, and do not initiate the 16-pair untouched
confirmation. Any next experiment requires an **independently justified,
published** editing method or externally anchored hyperparameter recipe as a
**new arm** before seeing its outcomes. First check that real free-generation
corrections are nonzero on development data, then evaluate sequential Q0/Q4/Q8
collision prevalence and matched next-edit consequences with the frozen data
manifest. If all strong methods fail, the current model/task may be unsuitable;
retain a bounded negative report instead of manufacturing a new operator.

No journal target selected, no manuscript or private artifacts uploaded.
