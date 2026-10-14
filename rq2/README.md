# RQ2: Cleanliness-Oriented Fine-Tuning

RQ2 investigates how cleanliness-based training data
selection affects code structure and functional correctness.

## Training Data

Six language-specific datasets are prepared:

| Language | 25 | 50 | 75 |
|----------|----|----|----|
| Python | py25.json | py50.json | py75.json |
| C++ | cpp25.json | cpp50.json | cpp75.json |

The labels 25, 50, and 75 refer to the
cleanliness-improvement selection thresholds.

## Model

Base model:
Qwen/Qwen2.5-Coder-7B-Instruct

Fine-tuning method:
LoRA

Training configurations are stored in ../configs/.

## Availability

The datasets and trained LoRA adapters are intended
for separate archival releases on Zenodo, subject
to dataset provenance and redistribution checks.
