# Tredence AI Engineer Case Study

## Self-Pruning Neural Network on CIFAR-10

This repository contains a complete implementation of the **Self-Pruning Neural Network** case study for the Tredence AI Engineering internship process.

## Objective

Build a neural network that learns to suppress unimportant connections during training by introducing learnable gates on weights and optimizing a combined objective:

$$
\mathcal{L}_{total} = \mathcal{L}_{classification} + \lambda \cdot \mathcal{L}_{sparsity}
$$

where sparsity is encouraged via L1-style regularization over gate activations.

## Implementation Summary

Primary implementation: [self_pruning_cifar10.py](self_pruning_cifar10.py)

Core components:
- Custom `PrunableLinear` layer with trainable `gate_scores`
- Gate transform via sigmoid and element-wise gated weights in forward pass
- End-to-end CIFAR-10 training pipeline with data augmentation
- Multi-`lambda` experiment runner
- Evaluation for test accuracy and sparsity percentage
- Automatic artifact generation (CSV, JSON, Markdown report, histogram)

## Repository Structure

- [self_pruning_cifar10.py](self_pruning_cifar10.py) — training, evaluation, reporting
- [requirements.txt](requirements.txt) — Python dependencies
- [deliverables/CASE_STUDY_REPORT.md](deliverables/CASE_STUDY_REPORT.md) — final submission report
- [deliverables/best_model_gate_histogram.png](deliverables/best_model_gate_histogram.png) — final gate distribution plot

## Environment Setup

1. Create/activate a Python virtual environment
2. Install dependencies:

`pip install -r requirements.txt`

## Run Experiments

Example command:

`python self_pruning_cifar10.py --epochs 5 --batch-size 128 --lambdas 0 1e-6 5e-6 --output-dir outputs_final`

Generated outputs:
- `results.csv`
- `results.json`
- `report.md`
- `best_model_gate_histogram.png`

## Reproducibility Notes

- Fixed seeding is used for deterministic behavior where possible.
- Results can vary across hardware and runtime settings.
- The `lambda` sweep controls the sparsity-versus-accuracy tradeoff.

## Deliverables for Submission

- Case-study report: [deliverables/CASE_STUDY_REPORT.md](deliverables/CASE_STUDY_REPORT.md)
- Gate histogram: [deliverables/best_model_gate_histogram.png](deliverables/best_model_gate_histogram.png)

## License

Submitted as a case-study assignment for evaluation purposes.
