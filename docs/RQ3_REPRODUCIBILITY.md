# RQ3: Reliability-Aware Self-Distillation

## Overview

RQ3 refines the cleanliness-oriented model from RQ2
using reliability-aware self-distillation.

The workflow consists of:

1. Candidate generation
2. Static reliability screening
3. Reliability-aware candidate selection
4. Conservative original-code fallback
5. LoRA fine-tuning
6. Functional correctness and cleanliness evaluation

## 1. Candidate Generation

The RQ2 checkpoint is used to generate four candidate
implementations per input.

The model is served through a LLaMA-Factory-compatible
OpenAI API endpoint using the Hugging Face backend.

Generation settings:

| Parameter | Value |
| --- | --- |
| Backbone | Qwen2.5-Coder-7B-Instruct |
| Adapter | RQ2 cleanliness-tuned LoRA |
| Temperature | 1.5 |
| Top-p | 0.8 |
| Top-k | 20 |
| Sampling | Enabled |
| Max new tokens | 1024 |
| Candidates per input | 4 |

The generation scripts do not explicitly override
sampling parameters in each API request.

The corresponding server configuration files are:

- configs/infer_QWcodemix25_rq3_enhanced_gen.yaml
- configs/infer_QWcodemix25_rq3_enhanced_cpp_gen.yaml

Candidate generation scripts:

- rq3/generate_enhanced25_python570_n4.py
- rq3/generate_enhanced25_cpp1000_n4.py

## 2. Static Reliability Screening

Python and C++ candidates are processed by
language-specific static analyses.

The screening procedures examine contract-related
properties and behavioral-footprint constraints.

Scripts:

- rq3/filter_enhanced25_python570_n4.py
- rq3/filter_enhanced25_cpp1000_n4.py

Static checks are risk-reduction mechanisms and
do not formally guarantee semantic equivalence.

## 3. Candidate Selection

Candidates satisfying the required screening
conditions are ranked lexicographically by:

1. Length-change risk
2. Whether the implementation remains unchanged
3. Number of non-empty code lines
4. Candidate identifier

The preferred character-length ratio interval is
[0.35, 1.20], with progressively higher risks
assigned to larger deviations.

The length ratio affects ranking rather than
acting as an absolute rejection threshold.

If no candidate passes the required checks,
the original implementation is used as a
conservative fallback training target.

Dataset construction scripts:

- rq3/build_enhanced25_strict570.py
- rq3/build_enhanced25_cpp_strict1000.py

## 4. Final Training Dataset

| Language | Selected candidates | Original anchors | Total |
| --- | ---: | ---: | ---: |
| Python | 375 | 195 | 570 |
| C++ | 398 | 602 | 1000 |
| Total | 773 | 797 | 1570 |

The final dataset contains 1,570 instruction/input/output
records.

Selection metadata are released separately with
the dataset archive.

## 5. Model Fine-Tuning

The RQ3 model is trained by continuing LoRA
fine-tuning from the RQ2 checkpoint.

Training configuration:

configs/train_rq3_enhanced25_strict1570.yaml

Both adapters use LoRA rank 8 and alpha 16.

## 6. Evaluation

The trained model is evaluated using lower-temperature
decoding (temperature 0.6).

Benchmarks include:

- HumanEval and HumanEval+
- MBPP and MBPP+
- BigCodeBench Full and Hard

## 7. Reproduction Notes

The original experiment configurations contain
absolute filesystem paths.

Researchers must adapt model, dataset, and output paths
for their own computing environments.

The dataset and model archives will be published
separately through Zenodo.

DOI information will be added after publication.
