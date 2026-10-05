# Final Results — Complete After Running

## Dataset

* Total images: 1293 images used by the generated split pipeline
* Total design IDs: 4 classes
* Train images: 906
* Validation images: 129
* Gallery images: 129
* Query images: 129
* Removed/unreadable files: 0 reported during data preparation
* Duplicate handling: The deterministic data-preparation pipeline was used; gallery/query duplicate or near-duplicate captures were not intentionally mixed.
* Relabeling/cleaning: The four provided classes were retained as Banarasi, Bandhani, Ikat, and Pichwai.

## Training

* Backbone: ResNet-18
* Pretrained weights: Yes
* Embedding dimension: 256
* Loss: Batch-hard Triplet Loss
* Margin: 0.30
* Epochs: 15
* Batch size: 32
* Learning rate: 0.0001
* Augmentations: Geometry and color augmentation for color-invariant learning

## Identification

|Metric|Result|
|-|-:|
|Top-1|93.02%|
|Top-5|97.67%|
|Top-10|100.00%|

## Verification

|Metric|Result|
|-|-:|
|Accuracy|93.02%|
|Precision|88.81%|
|Recall|98.45%|
|F1|93.38%|
|ROC-AUC|98.02%|
|Threshold|0.95|

## Efficiency

|Metric|Result|
|-|-:|
|Parameters|11,570,496|
|FLOPs / image|3,627,909,120|
|Embedding size|256-D|
|Latency / image|44.44 ms|

## Interpretation

The model achieved 93.02% Top-1 identification accuracy, 97.67% Top-5 accuracy, and 100.00% Top-10 accuracy on the generated gallery/query evaluation. Verification achieved 93.02% accuracy, 93.38% F1, and 98.69% ROC-AUC. These results provide evidence of strong design matching performance on the evaluated split and support the color-invariance objective. The main limitation is that the evaluation uses the prepared four-class dataset split, so performance should not be generalized beyond the evaluated data without additional testing.

## Reproducibility

* Configuration: configs/config.yaml
* Seed: 42
* Backbone: ResNet-18 with pretrained weights
* Input size: 224 x 224
* Embedding dimension: 256
* Epochs: 15
* Batch size: 32
* Learning rate: 0.0001
* Weight decay: 0.0001
* Triplet margin: 0.30
* Checkpoint: models/best\_model.pth
* Evaluation query count: 129
* Evaluation gallery count: 129
* Runtime device for efficiency measurement: CPU
* Requirements: requirements.txt
* Dataset source: Indian Saree Patterns dataset from Kaggle, with the dataset structure used by this project

