# Published-editor arm feasibility, without fitting science to the editor

Date: 2026-10-11. This is a **source inspection**, not a model execution. The original Qwen2.5-0.5B LoRA Gate A block and untouched confirmation are unchanged. This memo does not authorize new paid GPU execution.

## Verified published reference, not invented parameters

Published baseline: Yang et al., ACL 2025, The Mirage of Model Editing.
Code: https://github.com/WanliYoung/Revisit-Editing-Evaluation
Fixed inspected source ref: 61ccca3c1ecdb3976c9c0924e9352f3e64cc7d48

- FT-M Qwen2.5-7B: source hparams/FT/qwen2.5-7b.yaml, blob SHA 6a0a0adcbdb8c58c0abb199d8fda7ddc8f429c6. Method FT, target_new objective, layer 27 MLP down projection, 25 steps, learning rate 5e-4, max_length 40, batch 1, zero weight decay. This is on **7B, not 0.5B**, and has author-specific local model path.
- FT-M Llama3.2-3B: source hparams/FT/llama3.2-3b.yaml, blob SHA c90edd5f5cefea9519f03265176b56a93f4ef9d8. 25 steps, learning rate 5e-4, layer 21, max_length 40. This is a published, lower-parameter alternative subject to access and hardware eligibility, not a presumed equivalent reproduction.
- WISE Qwen2.5-7B: source hparams/WISE/qwen2.5-7b.yaml, blob SHA 30c21d2b2b7f9092b7a9cbc35e1d99dbb3c6385a. Different lifelong editor with 70 update iterations and routing/memory settings. Not a drop-in CPU Qwen 0.5B control.
- NSE ACL 2025 official repository (jianghoucheng/NSE) says at least one A40 48GB GPU and older software stack. Do not assume this can be reproduced on a stock GitHub-hosted CPU.
- The published reference edit.py ordinarily takes the first K data rows; its QAEdit route uses only locality[0]. Running it unchanged would conflict with our connected-subject split and all-five-locality evaluation. A controlled adapter would require an independently audited patch with diff and digest.
- Published WILD uses question-only context by default with natural stop/free generation. The earlier Qwen chat-template test was WILD-aligned but not an exact reproduction.

## Decision for the NEXT arm, not a new algorithm

**External candidate:** published FT-M, on a model and editing stack the source actually supports. Prefer Qwen2.5-7B as the primary reference and Llama3.2-3B as a compute-feasible alternative if available. Do not retune blocked Qwen 0.5B LoRA after observing results.

**Structural comparator:** WISE if resources and compatibility are independently established. Two distinct editing mechanisms are needed eventually to separate a finding about future-state consequences from a quirk of one editor.

**Transfer candidate:** CORe50 or CLEAR visual continual learning; verify a real and already published incremental learning baseline before activation. A second domain does not automatically create novelty.

## Pre-execution hard gates

1. Pin checkpoint commit, license, availability, memory/compute envelope, adapter compatibility, all source hashes and exact evaluation logic.
2. Use development data only to establish that the editor makes nonzero **free-generated** factual corrections. No opened confirmation while repairing viability. Preserve the 0/8 older Qwen LoRA failure.
3. New ARM_ID and frozen method/model/split/evaluation contract; no silent replacement of Q0/Q4/Q8 or the 16 untouched original confirmation pairs.
4. Compare to pre-action UQ/semantic perturbations, PSR-style history/test predictor, legal behavioral queries, calibrated confidence, no-probe baseline. Cloning/replay and gradients are costly, privileged oracles, not budget-matched black-box methods.
5. Require a real pre-action decision (execute/defer/route) and a groundable future task loss. Otherwise descriptive observed collisions are not a new scientific contribution.
6. Stop with NEAR_NEIGHBOR_SATURATED if existing baselines address exactly this decision at equal legal information/cost. Stop with INFRASTRUCTURE_UNAVAILABLE if published model/editor cannot be run lawfully and reproducibly.

## Primary source links

- https://aclanthology.org/2025.acl-long.745/
- https://github.com/WanliYoung/Revisit-Editing-Evaluation/tree/61ccca3c1ecdb3976c9c0924e9352f3e64cc7d48/hparams/FT
- https://github.com/WanliYoung/Revisit-Editing-Evaluation/tree/61ccca3c1ecdb3976c9c0924e9352f3e64cc7d48/hparams/WISE
- https://github.com/jianghoucheng/NSE
- https://proceedings.mlr.press/v306/ma26f.html
- https://proceedings.mlr.press/v78/lomonaco17a.html
- https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/2838023a778dfaecdc212708f721b788-Abstract-round2.html
