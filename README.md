# DeepLure — Color-Invariant Saree Design Recognition

This repository is a submission-oriented implementation of the DeepLure AIE-CASE assignment.

## Objective
Identify a saree from its surface design independently of the color palette, and support:
1. Identification — rank a gallery by similarity.
2. Verification — decide whether two images carry the same design.

## Requirement coverage
| DeepLure requirement | Implementation |
|---|---|
| 500-character approach note | `APPROACH_NOTE.txt` |
| Pre/post-processing pipeline | `src/dataset.py`, `src/inference.py` |
| Training strategy | `src/train.py` |
| Loss function | `src/losses.py` (batch-hard triplet loss) |
| Sampling | `src/dataset.py` (balanced class sampler) |
| Augmentations | `src/dataset.py` (geometry + color robustness) |
| End-to-end PyTorch | `src/` + `notebooks/DeepLure_Saree_Recognition.ipynb` |
| Identification | `src/inference.py` |
| Verification | `src/inference.py` |
| Gallery/query split | `src/dataset.py` |
| Evaluation | `src/evaluate.py` |
| Efficiency bonus | `src/efficiency.py` |
| Reproducibility | `requirements.txt`, `configs/config.yaml`, fixed seeds |
| Dataset disclosure | `DATASET.md` |

## Important dataset restriction
The DeepLure PDF states that the Drive corpus is proprietary to third-party vendors, must not be redistributed, and the copy must be deleted after the exercise. Therefore the dataset itself is NOT included in this ZIP.

Download/access the datasets separately:
- DeepLure corpus: https://drive.google.com/open?id=1V_DcXJ50QYV9nEqvq7fqKno_lhbSPdBM
- Indian Saree Patterns: https://www.kaggle.com/datasets/div456/indian-saree-patterns

## Expected data layout

The simplest supported layout is:

data/raw/
  design_0001/
    image1.jpg
    image2.jpg
  design_0002/
    image1.jpg
    image2.jpg

The folder name is treated as the design ID.

You may also prepare a CSV with columns:
`path,design_id`

## Run

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Prepare data
Put images under `data/raw/<design_id>/...`.

### 3. Create splits
```bash
python -m src.dataset --config configs/config.yaml
```

This creates:
`data/processed/train.csv`
`data/processed/val.csv`
`data/processed/gallery.csv`
`data/processed/query.csv`

### 4. Train
```bash
python -m src.train --config configs/config.yaml
```

### 5. Evaluate
```bash
python -m src.evaluate --config configs/config.yaml --checkpoint models/best_model.pth
```

### 6. Efficiency report
```bash
python -m src.efficiency --config configs/config.yaml --checkpoint models/best_model.pth
```

## Notebook
Open `notebooks/DeepLure_Saree_Recognition.ipynb` in Kaggle and update the data path. The notebook mirrors the end-to-end workflow.

## Method
A pretrained torchvision backbone is used as a feature extractor. A projection head produces normalized design embeddings. Training uses batch-hard triplet loss. Color transformations are deliberately strong enough to discourage reliance on hue/saturation while preserving the textile structure.

The model is used for retrieval with cosine similarity and verification with a threshold learned on the validation pairs.

## Do not fabricate results
The repository contains evaluation code and result templates, but it does not claim accuracy before the actual datasets are trained and evaluated. Run the notebook/scripts and paste the generated metrics into the final submission.

## Submission checklist
- [ ] Approach note reviewed and <= 500 characters
- [ ] Dataset source and modifications documented
- [ ] PyTorch training completed
- [ ] Gallery/query split documented
- [ ] Identification results reported
- [ ] Verification results reported
- [ ] Best checkpoint saved
- [ ] Parameter count / embedding size / latency reported if using efficiency bonus
- [ ] All pretrained weights and external data disclosed
- [ ] Notebook/repository access tested
- [ ] DeepLure proprietary corpus not redistributed


## Full requirement audit

See `REQUIREMENT_TRACEABILITY.md` for a one-to-one mapping from the assignment to this repository.

## Efficiency
Run:
```bash
python -m src.efficiency --config configs/config.yaml --checkpoint models/best_model.pth
```
This reports parameter count, embedding size, empirical latency, and profiler-estimated FLOPs where supported.

## Final validation
```bash
python validate_project.py
```

## Important distinction
The code and documentation can be prepared without the proprietary dataset, but **actual performance results cannot honestly be filled in until the supplied dataset is run**. This repository intentionally avoids fabricated results.
