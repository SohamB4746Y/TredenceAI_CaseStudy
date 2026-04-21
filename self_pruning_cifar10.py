from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_shift = -2.0

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

        nn.init.constant_(self.gate_scores, 2.0)

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(3.0 * (self.gate_scores + self.gate_shift))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated_weight = self.weight * self.gates()
        return F.linear(x, gated_weight, self.bias)


class PrunableMLP(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc1 = PrunableLinear(128 * 4 * 4, 256)
        self.fc2 = PrunableLinear(256, 128)
        self.fc3 = PrunableLinear(128, num_classes)
        self.dropout = nn.Dropout(0.35)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def prunable_layers(self) -> Iterable[PrunableLinear]:
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

    def sparsity_loss(self) -> torch.Tensor:
        return sum(layer.gates().sum() for layer in self.prunable_layers())

    def collect_gate_values(self) -> torch.Tensor:
        all_gates = [layer.gates().detach().flatten() for layer in self.prunable_layers()]
        return torch.cat(all_gates)


@dataclass
class ExperimentResult:
    lambda_value: float
    test_accuracy: float
    sparsity_percent: float


def build_dataloaders(data_dir: Path, batch_size: int, num_workers: int = 2) -> tuple[DataLoader, DataLoader]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=test_transform)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def train_one_epoch(
    model: PrunableMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lambda_sparse: float,
    device: torch.device,
    label_smoothing: float,
) -> tuple[float, float, float]:
    model.train()
    total_loss, total_cls, total_sparse = 0.0, 0.0, 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)

        cls_loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        sparse_loss = model.sparsity_loss()
        loss = cls_loss + lambda_sparse * sparse_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_cls += cls_loss.item() * images.size(0)
        total_sparse += sparse_loss.item() * images.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_cls / n, total_sparse / n


@torch.no_grad()
def evaluate(model: PrunableMLP, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def compute_sparsity(model: PrunableMLP, threshold: float = 1e-2) -> float:
    gates = model.collect_gate_values()
    sparse = (gates < threshold).sum().item()
    total = gates.numel()
    return 100.0 * sparse / max(total, 1)


def save_gate_histogram(gates: torch.Tensor, output_path: Path, bins: int = 100) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(gates.cpu().numpy(), bins=bins)
    plt.title("Distribution of Final Gate Values")
    plt.xlabel("Gate value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_results_csv(results: list[ExperimentResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda", "test_accuracy", "sparsity_percent"])
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "lambda": r.lambda_value,
                    "test_accuracy": f"{r.test_accuracy:.2f}",
                    "sparsity_percent": f"{r.sparsity_percent:.2f}",
                }
            )


def write_results_json(results: list[ExperimentResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in results]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_report(results: list[ExperimentResult], best_lambda: float, plot_rel_path: str, output_md: Path) -> None:
    rows = "\n".join(
        f"| {r.lambda_value:.1e} | {r.test_accuracy:.2f} | {r.sparsity_percent:.2f} |" for r in results
    )

    report = f"""# Self-Pruning Neural Network — Short Report

## Why L1 penalty on sigmoid gates encourages sparsity
Each trainable gate is computed as $g = \\sigma(s)$ where $g \\in (0,1)$ and $s$ is the learnable gate score.
The sparsity term uses:

$$
\\mathcal{{L}}_{{sparsity}} = \\sum_i g_i
$$

Since all $g_i$ are non-negative, minimizing this term pushes many gates toward values near 0.
When a gate approaches 0, the effective weight $w_i \\cdot g_i$ is suppressed, which behaves like pruning.
Combined with classification loss, optimization keeps only connections useful for prediction.

## Results (CIFAR-10)

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|---:|---:|---:|
{rows}

## Best model and gate distribution
Best model selected by highest test accuracy among tested lambdas: **$\\lambda = {best_lambda:.1e}$**.

Gate histogram:

![Gate Histogram]({plot_rel_path})
"""
    output_md.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a self-pruning MLP on CIFAR-10")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"), help="Directory for CIFAR-10")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"), help="Where to save artifacts")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per lambda")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Optimizer weight decay")
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 1e-6, 5e-6],
        help="List of lambda values to evaluate",
    )
    parser.add_argument("--sparsity-threshold", type=float, default=1e-2, help="Gate threshold for sparsity metric")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing value")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = build_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    all_results: list[ExperimentResult] = []
    best_acc = -1.0
    best_lambda = None
    best_gates = None

    for lam in args.lambdas:
        print(f"\n=== Training with lambda={lam:.1e} ===")
        model = PrunableMLP().to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        for epoch in range(1, args.epochs + 1):
            total_loss, cls_loss, sparse_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                lam,
                device,
                args.label_smoothing,
            )
            test_acc = evaluate(model, test_loader, device)
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:02d}/{args.epochs} | total={total_loss:.4f} "
                f"cls={cls_loss:.4f} sparse={sparse_loss:.4f} lr={current_lr:.6f} | test_acc={test_acc:.2f}%"
            )

        final_acc = evaluate(model, test_loader, device)
        final_sparsity = compute_sparsity(model, threshold=args.sparsity_threshold)

        result = ExperimentResult(
            lambda_value=lam,
            test_accuracy=final_acc,
            sparsity_percent=final_sparsity,
        )
        all_results.append(result)

        print(f"Final for lambda={lam:.1e}: acc={final_acc:.2f}% | sparsity={final_sparsity:.2f}%")

        if final_acc > best_acc:
            best_acc = final_acc
            best_lambda = lam
            best_gates = model.collect_gate_values().cpu()

    assert best_lambda is not None and best_gates is not None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "results.csv"
    json_path = args.output_dir / "results.json"
    plot_path = args.output_dir / "best_model_gate_histogram.png"
    report_path = args.output_dir / "report.md"

    write_results_csv(all_results, csv_path)
    write_results_json(all_results, json_path)
    save_gate_histogram(best_gates, plot_path)
    generate_report(all_results, best_lambda, plot_path.name, report_path)

    print("\nSaved artifacts:")
    print(f"- {csv_path}")
    print(f"- {json_path}")
    print(f"- {plot_path}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()
