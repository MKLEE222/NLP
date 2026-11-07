# Public-reference supplement, 2026-11-07

Scope: reproduce audited components of saddamaditya/name-that-llm-two-stage-lora-hungarian v1 (public LB0.604), not the full ensemble. Full source uses multiple8B/14B adapters, frozen backbones, external datasets and fitted blend weights. Those full dependencies are not present locally. The stronger alexchilton notebook source API returned404; its score is not reproduced.

Use the exact same five outer prompt folds as iteration2 (seed20261110) and reuse the frozen V2 OOF predictions for same-fold comparison. All results are internal research evaluation on previously exposed training data. No tuning on outer scores.

Fixed components:

- P0: public StyleGrams four TF-IDF views and LinearSVC C0.5 (source definitions, weights, min_df and random_state42 unchanged), official fold training only.
- P1: same StyleGrams with official fit plus version2 seyitkaangunes/ntl-extra-api-passages, CC BY4.0,4998 Claude passages. Use half of each covered class's official weight for original fit rows and half for external rows; uncovered classes retain original weight. Class totals therefore stay equal to original official class counts. Deduplicate and remove normalized exact overlaps with the complete official corpus without consulting labels. No external rows are evaluated as official validation.
- P2: P1's public prompt-centred twin. Fit centred official and external training groups, infer evaluation groups from evaluation text only with public normalized content average linkage threshold0.90; replace plain scores only on inferred groups of size2–18. This does not impose a one-per-label assignment. This single-source external ablation deliberately retains five-label groups; full notebook requires12+ labels after joining several datasets, so P1/P2 are component adaptations.
- Q0: frozen local Qwen2.5-7B-Instruct AutoModel (identified from local model card) truncated at20 layers; mean pools at layers6 and20, max_length512, mask padding, original decoder BOS convention. Use NF4 double quantization with bfloat16 compute to fit available12GB GPU, rather than source fp16. No LoRA or generation. StandardScaler and balanced LR C0.0025 fit on official fold training only. The instruction-tuned local checkpoint and quantization are explicit adaptations rather than the public base model.
- Q1: the same Q0 feature model with official training prompt centring and evaluation centring only on inferred groups of size2–18; otherwise Q0 predictions.
- E: fixed50/50 probability mean of P1 and Q0. No weight search or per-class gate.

Extract Qwen features once without labels, cache with data/config/index/protocol identifiers; fit heads per fold. If feature extraction fails, report Q/E blocked and still run P candidates. No synthetic feature substitution. No hidden test, no public/private LB probing, no extra API text generation. Preserve iterations1/2 and the CCC source. Report exact-source components separately from adaptations and full-reproduction blockers.

Primary descriptive comparison: each fixed candidate's pooled OOF balanced accuracy versus saved V2. Prompt-cluster paired bootstrap2000 draws seed20261201; report95% intervals and Bonferroni one-sided lower percentile0.05/6 for the six candidates. No candidate is selected as an unbiased final winner from this already-exposed corpus. Also report grouping ARI/coverage and external per-class counts/weight audit. Save row predictions and cached features locally only.
