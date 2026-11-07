# Name That LLM × CCC: frozen small kill test (2026-11-07)

## Source and scope

Official `train.csv` only. No external passages, hidden test labels, or leaderboard tuning. The original CCC project at `<private frozen CCC source>` is read-only. This is a new task adapter; the CCC compiler is not claimed to be a text classifier.

## Split and primary endpoint

Sort unique `prompt_id`, shuffle with NumPy `default_rng(20261107)`, allocate 25 groups to lockbox, 20 to development, and the rest to fit. No prompt appears in two partitions. All vectorizers, projections, scalers, pair models, centroids and classification models fit on the fit partition. Development selects one candidate through the predeclared gate; the lockbox is evaluated once. Primary endpoint is 18-class balanced accuracy on lockbox. Development scores are exploratory. Report every class recall and prompt-cluster bootstrap intervals.

## Fixed candidates

* A0: character TF-IDF (2–5), word TF-IDF (1–2) and 15 text-style features, each with balanced multinomial logistic regression. Average their class probabilities with weights 0.45/0.35/0.20. This is the strong public-baseline family, not a reproduction of any one contestant's exact code.
* A1: fit 96-dimensional SVD on fit-only word TF-IDF; learn a same-source pair score from same-model/different-prompt positive pairs and same-prompt/different-model negative pairs. Form a class-prototype score and blend 25% with A0. This tests the earlier BALA-inspired stay/change idea, not BALA's original operator.
* A2: fit a style-feature logistic model on within-prompt residuals in fit; infer prompt groups from validation text alone by word-TF-IDF average-linkage clustering into `ceil(n/18)` groups; blend 25% with A0. True prompt IDs are used only for a diagnostic upper bound. This tests the earlier response-residual idea, not TACIT's original operator.
* A3: Hungarian one-per-label assignment on A2 probabilities within recovered groups **only** when group size is exactly 18 and mean within-group cosine exceeds 0.25. Also report oracle-prompt assignment as a nondeployable diagnostic. This tests the earlier SlotShift-inspired joint decision, not the SlotShift algorithm.

## CCC adoption gate

The registered claim is "candidate X improves balanced accuracy over A0 on unseen prompt groups." For each of A1–A3, bootstrap development prompt groups 1,000 times with seed 20261108 and compute paired balanced-accuracy differences. A candidate has authority to replace A0 only if the 1.667th percentile of its paired bootstrap differences (a one-sided Bonferroni-adjusted 98.33% lower bound) exceeds zero. Choose the qualifying candidate with largest development difference; otherwise use A0. The gate's claim scope is candidate adoption; it does not certify individual predictions, Kaggle score, or calibration under distribution shift. This is a CCC-inspired task adapter, not a call to the frozen CCC core.

## Leakage checks and failure conditions

Assert split disjointness, feature fitting on fit only, valid 18 labels, unique IDs, and inspect the observed prompt-group sizes; the official data need not contain exactly 18 rows in every group. Recovered-group clustering may use the *unlabelled* evaluation batch only. Oracle groups may never feed the deployable A2/A3 scores or adoption gate. Record recovered-group ARI and assignment coverage. If training data is unavailable, leave results BLOCKED and do not substitute synthetic scores.
