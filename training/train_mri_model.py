"""
Install dependencies:
pip install scikit-learn numpy matplotlib pillow tqdm

Install CUDA-enabled PyTorch on Windows:
py -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126

Train an MRI brain tumor classifier with transfer learning.
Run with:
python training/train_mri_model.py
"""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_TRAIN_DIR = PROJECT_ROOT / "Data" / "Training"
PRIMARY_TEST_DIR = PROJECT_ROOT / "Data" / "Testing"
FALLBACK_TRAIN_DIR = PROJECT_ROOT / "Data" / "Dataset" / "Training"
FALLBACK_TEST_DIR = PROJECT_ROOT / "Data" / "Dataset" / "Testing"
FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "mri_model.pth"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "mri_model_best.pth"

BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
VALIDATION_SPLIT = 0.15
RANDOM_SEED = 42
IMAGE_SIZE = 224
NUM_CLASSES = 4
LABEL_SMOOTHING = 0.05
EARLY_STOPPING_PATIENCE = 4
FROZEN_BACKBONE_EPOCHS = 2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dataset_path(primary: Path, fallback: Path) -> Path:
    if primary.is_dir():
        return primary
    if fallback.is_dir():
        return fallback

    raise FileNotFoundError(
        "Could not find the dataset directory. Checked: "
        f"{primary} and {fallback}."
    )


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            # Keep anatomical structure intact to improve explainability localization.
            transforms.Resize((256, 256)),
            transforms.RandomAffine(
                degrees=7,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
                fill=0,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform


def create_split_indices(targets: list[int]) -> tuple[list[int], list[int]]:
    target_array = np.array(targets, dtype=np.int64)
    rng = np.random.default_rng(RANDOM_SEED)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for class_id in np.unique(target_array):
        class_indices = np.where(target_array == class_id)[0]
        rng.shuffle(class_indices)

        class_val_size = max(1, int(round(len(class_indices) * VALIDATION_SPLIT)))
        if class_val_size >= len(class_indices):
            class_val_size = len(class_indices) - 1

        if class_val_size <= 0:
            raise ValueError(
                "Validation split leaves no samples for training in at least one class."
            )

        val_indices.extend(class_indices[:class_val_size].tolist())
        train_indices.extend(class_indices[class_val_size:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    if not train_indices or not val_indices:
        raise ValueError("Failed to create a non-empty train/validation split.")

    return train_indices, val_indices


def create_dataloaders(
    train_dir: Path,
    test_dir: Path,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    train_transform, eval_transform = build_transforms()
    base_train_dataset = datasets.ImageFolder(train_dir)

    if len(base_train_dataset.classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, found {len(base_train_dataset.classes)}: "
            f"{base_train_dataset.classes}"
        )

    train_indices, val_indices = create_split_indices(base_train_dataset.targets)

    train_dataset = Subset(
        datasets.ImageFolder(train_dir, transform=train_transform),
        train_indices,
    )
    val_dataset = Subset(
        datasets.ImageFolder(train_dir, transform=eval_transform),
        val_indices,
    )
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    if base_train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError(
            "Training and testing folders must have the same class names. "
            f"Train classes: {base_train_dataset.classes}, "
            f"Test classes: {test_dataset.classes}"
        )

    pin_memory = torch.cuda.is_available()
    num_workers = min(4, os.cpu_count() or 1)
    use_persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers,
    )

    return train_loader, val_loader, test_loader, base_train_dataset.classes


def build_model(device: torch.device) -> nn.Module:
    try:
        if hasattr(models, "ResNet50_Weights"):
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            model = models.resnet50(pretrained=True)
    except Exception as error:
        raise RuntimeError(
            "Failed to load pretrained ResNet50 weights. Ensure torchvision is "
            "installed and the pretrained weights are available."
        ) from error

    model.fc = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(model.fc.in_features, NUM_CLASSES),
    )
    return model.to(device)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        if name.startswith("fc"):
            parameter.requires_grad = True
            continue
        parameter.requires_grad = trainable


def build_optimizer(model: nn.Module, learning_rate: float) -> AdamW:
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable model parameters found for optimizer initialization.")

    return AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )


def require_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this training script, but PyTorch cannot access a GPU. "
            "Install a CUDA-enabled PyTorch build, for example: "
            "py -m pip install --upgrade torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu128"
        )

    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    return torch.device("cuda")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        predictions = outputs.argmax(dim=1)
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (predictions == labels).sum().item()
        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{running_loss / total_samples:.4f}",
            acc=f"{running_correct / total_samples:.4f}",
        )

    epoch_loss = running_loss / total_samples
    epoch_accuracy = running_correct / total_samples
    return epoch_loss, epoch_accuracy


@torch.inference_mode()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        outputs = model(images)
        loss = criterion(outputs, labels)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (predictions == labels).sum().item()
        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{running_loss / total_samples:.4f}",
            acc=f"{running_correct / total_samples:.4f}",
        )

    epoch_loss = running_loss / total_samples
    epoch_accuracy = running_correct / total_samples
    _, _, macro_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )
    return epoch_loss, epoch_accuracy, float(macro_f1)


def save_best_checkpoint(
    model: nn.Module,
    class_names: list[str],
    best_val_accuracy: float,
    best_val_macro_f1: float,
) -> None:
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "best_val_accuracy": best_val_accuracy,
            "best_val_macro_f1": best_val_macro_f1,
            "image_size": IMAGE_SIZE,
            "imagenet_mean": IMAGENET_MEAN,
            "imagenet_std": IMAGENET_STD,
            "architecture": "resnet50",
        },
        BEST_MODEL_PATH,
    )


def save_final_checkpoint(
    model: nn.Module,
    class_names: list[str],
    best_val_accuracy: float,
    best_val_macro_f1: float,
) -> None:
    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "best_val_accuracy": best_val_accuracy,
            "best_val_macro_f1": best_val_macro_f1,
            "image_size": IMAGE_SIZE,
            "imagenet_mean": IMAGENET_MEAN,
            "imagenet_std": IMAGENET_STD,
            "architecture": "resnet50",
        },
        FINAL_MODEL_PATH,
    )


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float, np.ndarray]:
    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []

    progress_bar = tqdm(dataloader, desc="Testing", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=device.type == "cuda")
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu().tolist()

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )
    matrix = confusion_matrix(all_labels, all_predictions)

    return accuracy, precision, recall, f1_score, matrix


def main() -> None:
    set_seed(RANDOM_SEED)

    device = require_cuda_device()
    train_dir = resolve_dataset_path(PRIMARY_TRAIN_DIR, FALLBACK_TRAIN_DIR)
    test_dir = resolve_dataset_path(PRIMARY_TEST_DIR, FALLBACK_TEST_DIR)

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        train_dir,
        test_dir,
    )

    print(f"Using device: {device}")
    print(f"Training directory: {train_dir}")
    print(f"Testing directory: {test_dir}")
    print(f"Classes: {class_names}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    model = build_model(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    set_backbone_trainable(model, trainable=False)
    optimizer = build_optimizer(model, learning_rate=LEARNING_RATE * 2.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE * 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    best_val_score = -1.0
    best_val_accuracy = -1.0
    best_val_macro_f1 = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        if epoch == FROZEN_BACKBONE_EPOCHS + 1:
            set_backbone_trainable(model, trainable=True)
            optimizer = build_optimizer(model, learning_rate=LEARNING_RATE)
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, EPOCHS - FROZEN_BACKBONE_EPOCHS),
                eta_min=LEARNING_RATE * 0.1,
            )
            print("Unfroze ResNet backbone for full-model fine-tuning.")

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            True,
            device,
            epoch,
        )
        val_loss, val_accuracy, val_macro_f1 = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            epoch,
        )
        scheduler.step()

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        val_score = (0.70 * val_accuracy) + (0.30 * val_macro_f1)

        if val_score > best_val_score:
            best_val_score = val_score
            best_val_accuracy = val_accuracy
            best_val_macro_f1 = val_macro_f1
            best_weights = copy.deepcopy(model.state_dict())
            save_best_checkpoint(
                model,
                class_names,
                best_val_accuracy,
                best_val_macro_f1,
            )
            epochs_without_improvement = 0
            print(
                f"Saved best model to {BEST_MODEL_PATH} with validation accuracy "
                f"{val_accuracy:.4f} and macro-F1 {val_macro_f1:.4f}"
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                "Early stopping triggered after "
                f"{epochs_without_improvement} epoch(s) without validation improvement."
            )
            break

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best validation macro-F1: {best_val_macro_f1:.4f}")

    model.load_state_dict(best_weights)
    save_final_checkpoint(
        model,
        class_names,
        best_val_accuracy,
        best_val_macro_f1,
    )
    print(f"Saved final checkpoint copy to {FINAL_MODEL_PATH}")

    accuracy, precision, recall, f1_score, matrix = evaluate_model(
        model,
        test_loader,
        device,
    )

    print("\nTesting Metrics")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")
    print("Confusion Matrix:")
    print(matrix)


if __name__ == "__main__":
    main()