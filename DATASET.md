# Dataset and split protocol

## Sources required by the assignment
1. DeepLure saree corpus:
   https://drive.google.com/open?id=1V_DcXJ50QYV9nEqvq7fqKno_lhbSPdBM
2. Indian Saree Patterns (Kaggle):
   https://www.kaggle.com/datasets/div456/indian-saree-patterns

The assignment permits combining, re-splitting, cleaning, and relabeling, provided those actions are documented.

## Privacy / redistribution
Do not put the DeepLure proprietary corpus in GitHub or this ZIP. Delete your local copy when the exercise concludes, as required by the assignment.

## Recommended split
Split by design identity first, then images within each design:
- train: 70%
- validation: 10%
- gallery: 10%
- query: 10%

For each design with enough images, keep at least one gallery and one query image. Do not put duplicate/near-duplicate captures in both gallery and query.

The script uses a deterministic seed and attempts to keep every class represented.

## What to document after inspecting the actual data
Fill these values in the final report:
- number of designs
- number of images
- number of colorways
- broken/removed files
- duplicate policy
- final train/validation/gallery/query counts
- any relabeling performed

Do not invent these values before running the data-preparation script.
