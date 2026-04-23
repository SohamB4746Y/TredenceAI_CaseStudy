# Self-Pruning Neural Network — Case Study Report

## Executive summary
This case study implements a self-pruning neural network on CIFAR-10 using trainable gates that suppress less useful connections during learning. Instead of separating training and pruning into two stages, the model jointly learns predictive weights and gate activations under a combined objective. This creates a direct and differentiable mechanism for balancing accuracy and sparsity.

My approach focused on engineering rigor and interpretability: custom prunable layers, reproducible training setup, controlled lambda sweep, explicit sparsity metric, and artifact generation for analysis. The implementation uses a CNN feature extractor with prunable fully connected layers, SGD with momentum and cosine annealing, data augmentation, and label smoothing.

The final run achieved stable CIFAR-10 performance across tested lambda values while measured sparsity remained limited under the selected threshold. That result is still valuable: it validates the self-pruning training pipeline and highlights the next optimization direction, namely stronger sparsity pressure (higher lambda or schedule-based regularization) to induce more aggressive gate suppression.

## Problem statement and objective
The goal is to build a model that learns to identify and suppress unimportant connections during training using trainable gates. The optimization target is:

$$
\mathcal{L}_{total} = \mathcal{L}_{classification} + \lambda \cdot \mathcal{L}_{sparsity}
$$

where the sparsity term penalizes active gates and $\lambda$ controls the tradeoff between predictive quality and compactness.

## My understanding of the case study
I interpreted this problem as a tradeoff-learning exercise rather than pure accuracy maximization. A successful solution should:
- implement differentiable gating correctly,
- preserve training stability,
- report both accuracy and sparsity,
- and explain the outcome honestly, including limitations.

In this formulation, each prunable connection is scaled by a gate value in $(0,1)$. When optimization drives gates closer to zero, those connections become effectively inactive. This behaves like pruning without hard thresholding during gradient updates.

## Architecture and implementation approach
The implemented model has two parts:
- Convolutional backbone for feature extraction (Conv + BatchNorm + ReLU + MaxPool blocks).
- Classifier head with three custom `PrunableLinear` layers.

For each prunable layer, effective weights are computed as:

$$
W_{effective} = W \odot g
$$

where $g$ is obtained from learnable gate scores using a sigmoid transform. This keeps the mechanism smooth and trainable end-to-end.

## Training and evaluation setup
- Dataset: CIFAR-10
- Data transforms: random crop + horizontal flip (train), normalization (train/test)
- Optimizer: SGD with momentum and Nesterov
- Scheduler: cosine annealing learning rate
- Regularization supports: label smoothing, weight decay
- Reproducibility controls: fixed seed and deterministic settings where feasible

Tested lambda values:
- $0.0$
- $1.0 \times 10^{-6}$
- $5.0 \times 10^{-6}$

Reported metrics:
- Test accuracy (%)
- Sparsity level (%) based on fraction of gates below threshold $1\times10^{-2}$

## Key observations
- The self-pruning mechanism is correctly integrated and trainable.
- Accuracy remains consistent across the tested lambda range.
- Measured sparsity is minimal at current regularization strength.

This indicates that the pipeline works, but stronger sparsity pressure is needed to push more gates toward near-zero values under the chosen metric.

## Why L1 penalty on sigmoid gates encourages sparsity
Each gate is computed as $g = \sigma(s)$, where $g \in (0,1)$ and $s$ is a learnable score.
The sparsity term is:

$$
\mathcal{L}_{sparsity} = \sum_i g_i
$$

Because all $g_i$ are non-negative, minimizing this term drives many gates toward 0.
As gates shrink, effective weights $w_i \cdot g_i$ are suppressed, which behaves like pruning.
The classifier keeps only useful connections under the tradeoff controlled by $\lambda$.

## Results (Final CIFAR-10 run)

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|---:|---:|---:|
| 0.0e+00 | 67.66 | 0.00 |
| 1.0e-06 | 67.69 | 0.00 |
| 5.0e-06 | 67.76 | 0.00 |

## Gate value distribution (best model from final run)

![Gate Histogram](best_model_gate_histogram.png)
