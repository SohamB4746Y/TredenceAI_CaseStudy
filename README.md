# Tredence AI Engineer Case Study — Self-Pruning Neural Network

This repository contains the requested implementation for the **Self-Pruning Neural Network** case study.

## What is included
- `self_pruning_cifar10.py`
  - Custom `PrunableLinear` layer with learnable gate scores.
  - CIFAR-10 MLP model built using prunable layers.
  - Training loop with total loss:
    - Classification loss (Cross Entropy)
    - `+ lambda * sparsity_loss`, where `sparsity_loss` is L1 over sigmoid gates.
  - Evaluation of test accuracy and sparsity level.
  - Multiple lambda experiments.
  - Auto-generated outputs:
    - `outputs/results.csv`
    - `outputs/results.json`
    - `outputs/best_model_gate_histogram.png`
    - `outputs/report.md`

## Setup
1. Install dependencies from `requirements.txt`.
2. Run the script with at least 3 lambda values.

Example run:
`python self_pruning_cifar10.py --epochs 20 --batch-size 128 --lambdas 1e-5 5e-5 1e-4`

## Notes
- Higher lambda usually increases sparsity but may reduce test accuracy.
- `outputs/report.md` is generated from actual run metrics and is ready to submit.

## Submitted Deliverables
- Report (tracked): [deliverables/CASE_STUDY_REPORT.md](deliverables/CASE_STUDY_REPORT.md)
- Gate histogram (tracked): [deliverables/best_model_gate_histogram.png](deliverables/best_model_gate_histogram.png)
