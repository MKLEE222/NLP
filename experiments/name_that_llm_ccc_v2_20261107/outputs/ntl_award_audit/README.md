# Award-readiness audit

This audit reuses frozen outer models, runs batch-composition stress tests and a matched217-feature linear control, inspects error classes, and repairs deployment cache identity checks. It does not train another proposed classifier or produce a Kaggle score.

Read `AWARD_READINESS.md`. Key finding: the same2699 rows score57.46% as complete outer batches but42.30% in32-row batches and43.93% with singleton semantics. This is a real transductive dependency, not a prediction of the unknown private-test score. The active prediction entry point groups the entire input file; backbone encoding minibatches alone do not cause this drop.

The cached-feature bug was fixed in `../ntl_workspace_v4/predict.py` with `encoder_checks.py` and an artifact-specific `verified_encoder_manifest.json`. Actual model/tokenizer/source and feature hashes are checked before cache use. Wrong model/tokenizer/stamp tests and fresh3-text anchor replay passed. The binding script contains original-machine paths and anchor filenames; a portable training/deployment rebinding workflow and a Kaggle notebook remain outstanding. Frozen original scores, method and protocol are retained.

`stress_audit.py` and `matched_control.py` use the saved local research artifacts and original path defaults; they expose no passages in their aggregate outputs. No model binary, corpus, row predictions or feature cache is published here. The main report distinguishes conditional improvements, coding errors, weak controls, missing independent validation and award requirements.
