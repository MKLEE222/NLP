# Name That LLM: prompt-disjoint transfer kill test

2026-11-07. An exploratory source-attribution test on official competition training data. The CSV/ZIP is obtained separately through Kaggle after accepting competition rules. No competition passages or row predictions are distributed here.

The frozen split uses seed 20261107: 105 fit prompts (1,890 rows), 20 development prompts (360 rows), 25 lockbox prompts (449 rows). The input archive SHA-256 is recorded in results.json. All learned representations fit on the fit partition. There are no exact or normalized duplicate passages across partitions.

| Method | Development balanced accuracy | Lockbox balanced accuracy |
|---|---:|---:|
| A0: char/word/style probability blend | 0.31944 | 0.33120 |
| A1: SVD paired-source prototype blend | 0.24722 | 0.26000 |
| A2: recovered-prompt style residual blend | 0.31944 | 0.33343 |
| A3: conditional recovered-group assignment | 0.31944 | 0.32898 |
| B0: concatenated char/word/style LR, post hoc exploratory | 0.32500 | 0.36454 |

The predeclared claim-scoped gate retained A0: none of the three development paired-bootstrap lower bounds exceeded zero after adjustment. A1 degraded, A2 did not establish a benefit, and A3 had only 8.02% lockbox assignment coverage. A2's lockbox paired 95% interval versus A0 was [-0.03842, 0.04000]. This does not support a transfer gain.

The gate is inspired by CCC's contract and evidence-authority interface; the frozen CCC core is not invoked. A1/A2/A3 are proxies for suggestions in an earlier discussion, not implementations or tests of the original BALA/TACIT/SlotShift operators. B0 belongs to the public TF-IDF/style/LR family and is not an exact reproduction of any contestant's code. A pretrained or LoRA public baseline remains unreplicated.

The first run took about 12.8 seconds on CPU. A deterministic replay added reporting intervals, provenance and duplicate checks only; original scores and choices matched exactly. B0 was declared and run after the first lockbox opening, so is exploratory. Further tuning must treat this lockbox as already observed.

```sh
pip install -r requirements.txt
python run_killtest.py /path/to/name-that-llm.zip
python public_family_check.py /path/to/name-that-llm.zip
```

Use Python 3.13.7. The public protocol redacts only the private CCC source path; the result's protocol SHA-256 refers to the original local protocol. Original research files were not modified. The script SHA-256 in results.json covers the distributed primary script. Results include per-class recalls and prompt-cluster intervals.

Official sources: [competition data](https://www.kaggle.com/competitions/name-that-llm/data), [rules](https://www.kaggle.com/competitions/name-that-llm/rules). Public reference routes: [TF-IDF discussion](https://www.kaggle.com/competitions/name-that-llm/discussion/751802), [two-stage LoRA and Hungarian](https://www.kaggle.com/code/saddamaditya/name-that-llm-two-stage-lora-hungarian).
