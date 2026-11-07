# Iteration 2: stronger classifier and measured gate limits

Fixed five-fold nested prompt-group evaluation, seed20261110. All 150 prompts were previously exposed during research, so this is internal evaluation, not an untouched external test.

| Fixed model | OOF balanced accuracy |
|---|---:|
| B0 TF-IDF/style LR | 0.34114 |
| V1 richer-style LR | 0.34042 |
| V2 richer-style LinearSVC | **0.35929** |
| V3 frozen MiniLM blend | 0.34153 |
| Inner evidence-gate selected | 0.34152 |

V2 improves over same-fold B0 by 1.81 percentage points; paired prompt bootstrap95% interval [0.29,3.37] pp. It exceeds B0 on all five outer folds. The gate chooses B0 in four folds and V2 in one, with no established aggregate gain. The first-round single lockbox scores use a different partition and should not be compared directly.

An actual frozen CCC interval-backend call evaluated finite-cohort V1-versus-B0 auditing under the separately predeclared contract. Greedy disagreement auditing and100 random orders all needed150 prompt audits: no cost saving. The wrapper imports the user's existing frozen backend without edits; original backend source is not distributed. This is a component integration, not the full CCC compiler.

Directory layout preserves the local evaluation's `outputs` and `work` structure. From this directory's repository parent containing `outputs`, obtain the official competition ZIP separately and the public local MiniLM checkpoint revision1110a243fdf4706b3f48f1d95db1a4f5529b4d41, then:

```sh
pip install -r outputs/ntl_ccc_v2/requirements.txt
python outputs/ntl_ccc_v2/run_portable.py /path/to/name-that-llm.zip /path/to/local/MiniLM/checkpoint
python outputs/ntl_ccc_v2/core_audit.py /path/to/frozen/aggregate_claim_compiler.py
python outputs/ntl_ccc_v2/build_model.py /path/to/name-that-llm.zip V2
python outputs/ntl_ccc_v2/predict.py outputs/ntl_ccc_v2/V2_fulltrain.joblib input.csv output.parquet
```

Use Python3.13.7. The checkpoint card is [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Run `run_portable.py` to override the primary script's original machine-specific cache default. CUDA is optional; the final SVM predictor needs no embeddings or GPU.

The local fitted model is21.6MB and passed save/reload prediction equivalence. Model binary, competition data, sample predictions, row-level OOF predictions and cached embeddings are not uploaded. `model_manifest.json` describes the locally built artifact. Reports and measured summary results accompany the scripts. No Kaggle submission was made.
