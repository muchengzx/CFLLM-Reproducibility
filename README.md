# CFLLM: A Reliability Framework for Clean Function Generation with Large Language Models

This repository contains research artifacts for CFLLM,
including code cleanliness analysis, cleanliness-oriented
fine-tuning, reliability-aware self-distillation, and
benchmark evaluation.

## Research Questions

### RQ1: Evolution of Code Cleanliness

RQ1 investigates function-level structural cleanliness
evolution in Python and C++ open-source projects.

The analysis considers three metrics:

- Cyclomatic Complexity (CC)
- Function Length (Rows)
- Number of Parameters (Args)

Repository mining and historical data collection build
upon our previous work:

Dapeng Yan et al., "Evolving Trends in Cleanliness of
Open Source Projects," ACM Transactions on Software
Engineering and Methodology, 2026.

Prior project documentation:
https://github.com/blackantywj/CFLLM/tree/main/data_crawler

ECC dataset:
https://huggingface.co/datasets/vincent20000/ECC-Dataset

The CFLLM-specific RQ1 analysis artifacts are not yet
fully included in this release.

### RQ2: Cleanliness-Oriented Fine-Tuning

RQ2 studies the relationship between structural
simplification and functional correctness.

The experiment uses Python and C++ training datasets
selected using cleanliness-improvement thresholds
of 25%, 50%, and 75%.

The base model is:

Qwen/Qwen2.5-Coder-7B-Instruct

The repository includes training configuration templates
and related experimental scripts. These have not all been
verified as an end-to-end reproduction pipeline.

### RQ3: Reliability-Aware Self-Distillation

RQ3 introduces reliability-aware candidate selection
to balance structural cleanliness and functional
reliability.

The training data construction process includes:

1. Candidate generation
2. Static reliability and structural screening
3. Candidate selection and original-code fallback
4. Self-distillation fine-tuning

The final training dataset contains 1,570 examples:
570 Python examples and 1,000 C++ examples.

The RQ3 folder contains the candidate generation,
screening, and dataset construction scripts.

## Repository Structure

- rq1/: RQ1 documentation and analysis artifacts
- rq2/: RQ2 training and auxiliary scripts
- rq3/: RQ3 self-distillation pipeline
- evaluation/: Cleanliness and benchmark evaluation
- configs/: Training and inference configuration templates
- tools/: Configuration utilities
- docs/: Additional reproducibility documentation

## Evaluation

Evaluation scripts are included for:

- HumanEval
- MBPP
- BigCodeBench
- Code cleanliness metrics

Some evaluation scripts currently contain environment-
specific paths and require configuration before use.
A complete end-to-end reproduction has not been verified.

## Data and Model Availability

RQ2/RQ3 datasets and LoRA adapter checkpoints are
being prepared for separate Zenodo records.

Dataset redistribution permissions and release metadata
will be reviewed before public publication.

## License

The RQ3 source directory contains an MIT license
for the authorized RQ3 implementation.

Licensing of other components, third-party material,
datasets, and model artifacts must be assessed separately.

## Citation

Citation information for the CFLLM paper will be
updated when the final bibliographic record is available.
