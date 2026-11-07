# Public-reference components: 55.64% internal prompt CV

2026-11-07. Supplement to the earlier CCC migration kill test. Five identical outer prompt folds to iteration2; no new fold tuning. All official train data have previously influenced research decisions, so this is internal evaluation rather than an untouched final test.

| Fixed method | OOF balanced accuracy |
|---|---:|
| Saved iteration2 V2 SVM |0.35929|
| P0 public four-view StyleGrams |0.37042|
| P1 weighted external-data StyleGrams |0.38786|
| P2 recovered-group residual StyleGrams |0.43486|
| Q0 frozen Qwen independent head |0.41785|
| Q1 frozen Qwen recovered-group residual head |**0.55642**|
| E fixed P1/Q0 probability average |0.42452|

Q1 versus V2 paired prompt-bootstrap95% interval: +17.64 to+21.71 percentage points. It scores0.50581 versus independent Q0's0.43661 on a predeclared single-fold random-half batch;0.56799 versus0.49972 on a class-biased partial batch. These stress diagnostics are not new confirmation data.

The public0.604 Notebook's source was inspected, but its full ensemble of LoRA8B/14B, backbones, training outputs and external sources remains unreplicated. P0 uses the audited StyleGrams definitions. P1/P2 are single-source external-data adaptations. Q0/Q1 use an existing local **Qwen2.5-7B-Instruct**,20 frozen layers, pooled layers6/20, NF4 double quantization/bf16 compute and source-inspired heads; checkpoint and quantization differ from the public base model. Inferred groups come only from evaluation text; no true evaluation prompt IDs or one-per-label assignment are used. The improved score cannot be attributed to CCC.

This folder is intentionally placed next to the existing `ntl_ccc_v2` and `ntl_ccc_killtest` helpers in the iteration2 harness. First run iteration2 from its own instructions to create `work/v2_oof_predictions.npz`. Obtain official competition data separately and the external dataset version2 separately. From the harness root containing `outputs` and `work`:

```sh
pip install -r outputs/ntl_public_v3/requirements.txt
mkdir -p work/public_refs
curl -fL -o work/public_refs/extra_v2.zip 'https://www.kaggle.com/api/v1/datasets/download/seyitkaangunes/ntl-extra-api-passages?datasetVersionNumber=2'
python outputs/ntl_public_v3/extract_qwen.py /path/to/name-that-llm.zip /path/to/local/Qwen2.5-7B-Instruct
python outputs/ntl_public_v3/run_style.py /path/to/name-that-llm.zip work/public_refs/extra_v2.zip
python outputs/ntl_public_v3/run_qwen_heads.py /path/to/name-that-llm.zip
```

`run_stress.py` and its protocol contain original machine defaults; adjust only the data/backbone paths if moving machines. `build_qwen_heads.py` creates a local2.3MB head bundle requiring the matching local backbone. Its artifact and an actual3-row official sample inference were verified. No official data, external passages, row predictions, cached features or binaries are distributed here. Public code excerpts are attributed under Apache2.0; external data is CC BY4.0. See the notices, source manifest, protocols and report for provenance and the full-reproduction blockers.
