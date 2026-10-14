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

### Source Code

Source code and reproducibility documentation are maintained in this GitHub repository.

### Datasets

- [CFLLM RQ2/RQ3 datasets (Zenodo)](https://zenodo.org/records/23366406)
- Dataset DOI: [10.5281/zenodo.23366406](https://doi.org/10.5281/zenodo.23366406)

The Zenodo dataset archive contains the RQ2 Python/C++ training datasets and
RQ3 candidate-generation, screening, and self-distillation datasets.

### Model Checkpoints

- [CFLLM v0.1.0 LoRA adapter release](https://github.com/muchengzx/CFLLM-Reproducibility/releases/tag/v0.1.0)
- [Download CFLLM-Models.zip](https://github.com/muchengzx/CFLLM-Reproducibility/releases/download/v0.1.0/CFLLM-Models.zip)
- Base model: [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)

The GitHub release contains the RQ2 (`rq2_25`) and RQ3
(`rq3_strict1570`) LoRA adapters, not the full base model.

Redistribution and reuse of datasets, model artifacts, and third-party
materials remain subject to their applicable permissions and licenses.

## License

The original CFLLM source code in this repository is released under the
[MIT License](LICENSE) (copyright: CFLLM Contributors, 2026).

Third-party dependencies and source materials remain under their respective
licenses. The [Zenodo dataset](https://zenodo.org/records/23366406) and
[LoRA adapter release](https://github.com/muchengzx/CFLLM-Reproducibility/releases/tag/v0.1.0)
are separate research artifacts and are not automatically covered by the
source-code MIT license.

## Citation

Citation information for the CFLLM paper will be
updated when the final bibliographic record is available.
