# DeepLure Requirement Traceability

This file maps every explicit requirement in the supplied PDF to a project artifact.

| PDF requirement | Where it is covered | Final action |
|---|---|---|
| Identify design independent of color | README, model, color augmentation | Run and report results |
| Identification | `src/inference.py`, `src/evaluate.py` | Report Top-K |
| Verification | `src/inference.py`, `src/evaluate.py` | Report threshold + metrics |
| 500-character approach note | `APPROACH_NOTE.txt` | Keep at <=500 chars |
| Architecture rationale | README + approach note | Explain in interview |
| Complete pre-processing | `src/dataset.py` | Document actual data settings |
| Complete post-processing | `src/inference.py` | Document similarity/ranking |
| Training strategy | `src/train.py` | Report epochs/hyperparameters |
| Loss functions | `src/losses.py` | Explain triplet loss |
| Sampling | `src/dataset.py` | Explain positive/negative construction |
| Augmentations | `src/dataset.py` | Explain color robustness |
| End-to-end PyTorch | `src/`, notebook | Execute cleanly |
| Training + inference runnable | `src/train.py`, `src/inference.py` | Execute |
| Evaluation protocol | `src/evaluate.py`, DATASET.md | Justify split |
| Gallery/query split | `src/dataset.py` | Report actual counts |
| Parameter count | `src/efficiency.py` | Run |
| FLOPs and/or latency | `src/efficiency.py` | Run; report both where available |
| Embedding size | `configs/config.yaml` + efficiency output | Report 256-D |
| Lean/fast/novel architecture bonus | README + efficiency report | Defend choices |
| Dataset 1 disclosure | DATASET.md | Do not redistribute proprietary corpus |
| Dataset 2 disclosure | DATASET.md | Cite source |
| Data cleaning/relabeling | DATASET.md | Fill actual changes |
| PyTorch framework | requirements + source | Confirm |
| Kaggle compute | notebook | Run on Kaggle |
| Pretrained backbone disclosure | README | State ResNet-18 pretrained |
| External data disclosure | DATASET.md | State sources |
| Own-work / defense requirement | README | Be able to explain every decision |
| Notebook/repository submission | notebook + README | Verify reviewer access |
| Evaluation quality/reproducibility | README + config + seed | Run from clean environment |
| Code quality | modular `src/` | Run tests/checks |
| Clarity of reasoning | README + approach note | Final review |

## No-fabrication rule
The repository does not contain invented performance numbers. Actual metrics must be generated from the supplied datasets.

## Final evidence required before submission
1. `results/metrics.json`
2. `results/efficiency.json`
3. A completed dataset statistics section
4. Screenshots/outputs in the Kaggle notebook showing training and evaluation
5. A final README/results section with actual numbers
