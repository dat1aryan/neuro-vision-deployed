"""
Install dependencies:
pip install torch torchvision scikit-learn numpy matplotlib pillow tqdm pytorch-grad-cam opencv-python

Run with:
python training/gradcam_training.py
"""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from tqdm import tqdm

try:
    import cv2
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError as error:
    raise ImportError(
        "Missing GradCAM dependencies. Install with: "
        "pip install pytorch-grad-cam opencv-python matplotlib"
    ) from error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_TRAIN_DIR = PROJECT_ROOT / "Data" / "Training"
PRIMARY_TEST_DIR = PROJECT_ROOT / "Data" / "Testing"
FALLBACK_TRAIN_DIR = PROJECT_ROOT / "Data" / "Dataset" / "Training"
FALLBACK_TEST_DIR = PROJECT_ROOT / "Data" / "Dataset" / "Testing"
PRIMARY_TEST_IMAGE = PROJECT_ROOT / "test.jpg"
FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "mri_model.pth"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "mri_model_best.pth"
GRADCAM_OUTPUT_PATH = PROJECT_ROOT / "gradcam_output.jpg"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
VALIDATION_SPLIT = 0.15
RANDOM_SEED = 42
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
LABEL_SMOOTHING = 0.03
FROZEN_BACKBONE_EPOCHS = 2


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


def resolve_gradcam_image_path(test_dir: Path) -> Path:
    if PRIMARY_TEST_IMAGE.is_file():
        return PRIMARY_TEST_IMAGE

    for file_path in sorted(test_dir.rglob("*")):
        if file_path.suffix.lower() in IMAGE_EXTENSIONS:
            return file_path

    raise FileNotFoundError(
        "Could not find test.jpg in the project root or any image inside the testing "
        f"dataset at {test_dir}."
    )


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            # Preserve anatomy while still improving robustness.
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


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        if name.startswith("fc"):
            parameter.requires_grad = True
            continue
        parameter.requires_grad = trainable


def build_class_weights(labels: list[int], device: torch.device) -> torch.Tensor:
    labels_array = np.array(labels, dtype=np.int64)
    counts = np.bincount(labels_array, minlength=len(CLASS_NAMES)).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.sum() * len(CLASS_NAMES)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def create_dataloaders(
    train_dir: Path,
    test_dir: Path,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], transforms.Compose, list[int]]:
    train_transform, eval_transform = build_transforms()
    base_train_dataset = datasets.ImageFolder(train_dir)

    if base_train_dataset.classes != CLASS_NAMES:
        raise ValueError(
            "Training dataset classes must be exactly "
            f"{CLASS_NAMES}, found {base_train_dataset.classes}."
        )

    total_samples = len(base_train_dataset)
    validation_size = max(1, int(total_samples * VALIDATION_SPLIT))
    training_size = total_samples - validation_size

    if training_size <= 0:
        raise ValueError("Training split is empty. Add more images to the dataset.")

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = torch.randperm(total_samples, generator=generator).tolist()
    train_indices = indices[:training_size]
    val_indices = indices[training_size:]

    train_dataset = Subset(
        datasets.ImageFolder(train_dir, transform=train_transform),
        train_indices,
    )
    val_dataset = Subset(
        datasets.ImageFolder(train_dir, transform=eval_transform),
        val_indices,
    )
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    if test_dataset.classes != CLASS_NAMES:
        raise ValueError(
            "Testing dataset classes must be exactly "
            f"{CLASS_NAMES}, found {test_dataset.classes}."
        )

    pin_memory = torch.cuda.is_available()

    train_targets = [base_train_dataset.targets[index] for index in train_indices]
    class_counts = np.bincount(np.array(train_targets, dtype=np.int64), minlength=len(CLASS_NAMES)).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    sample_weights = [1.0 / float(class_counts[label]) for label in train_targets]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, base_train_dataset.classes, eval_transform, train_targets


def build_model(device: torch.device) -> nn.Module:
    try:
        if hasattr(models, "ResNet50_Weights"):
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            model = models.resnet50(pretrained=True)
    except Exception as error:
        raise RuntimeError(
            "Failed to load pretrained ResNet50 weights. Ensure torchvision is "
            "installed and the pretrained weights are available."
        ) from error

    model.fc = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(model.fc.in_features, len(CLASS_NAMES)),
    )
    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
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
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predictions = outputs.argmax(dim=1)
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (predictions == labels).sum().item()
        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{running_loss / total_samples:.4f}",
            acc=f"{running_correct / total_samples:.4f}",
        )

    return running_loss / total_samples, running_correct / total_samples


@torch.inference_mode()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        outputs = model(images)
        loss = criterion(outputs, labels)
        predictions = outputs.argmax(dim=1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (predictions == labels).sum().item()
        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{running_loss / total_samples:.4f}",
            acc=f"{running_correct / total_samples:.4f}",
        )

    return running_loss / total_samples, running_correct / total_samples


def prepare_image_for_gradcam(
    image_path: Path,
    transform: transforms.Compose,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_image = np.array(resized_image, dtype=np.float32) / 255.0
    input_tensor = transform(resized_image).unsqueeze(0).to(device)
    return input_tensor, rgb_image


def generate_gradcam_visualization(
    model: nn.Module,
    image_path: Path,
    transform: transforms.Compose,
    class_names: list[str],
    device: torch.device,
) -> tuple[str, float]:
    input_tensor, rgb_image = prepare_image_for_gradcam(image_path, transform, device)

    model.eval()
    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    target_layers = [model.layer3[-1].conv3, model.layer4[-1].conv3]
    targets = [ClassifierOutputTarget(predicted_index.item())]

    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam_pp:
        cam_primary = cam_pp(input_tensor=input_tensor, targets=targets)[0]
        flipped_cam = cam_pp(input_tensor=torch.flip(input_tensor, dims=[3]), targets=targets)[0]

    flipped_cam = np.flip(flipped_cam, axis=1)

    with GradCAM(model=model, target_layers=target_layers) as cam_std:
        cam_regular = cam_std(input_tensor=input_tensor, targets=targets)[0]

    grayscale_cam = (0.55 * cam_primary) + (0.25 * flipped_cam) + (0.20 * cam_regular)
    grayscale_cam = np.clip(grayscale_cam, 0.0, None)
    if float(grayscale_cam.max()) > 1e-8:
        grayscale_cam = grayscale_cam / float(grayscale_cam.max())

    cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    cv2.imwrite(str(GRADCAM_OUTPUT_PATH), cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

    predicted_class = class_names[predicted_index.item()]
    return predicted_class, confidence.item()


def main() -> None:
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_dir = resolve_dataset_path(PRIMARY_TRAIN_DIR, FALLBACK_TRAIN_DIR)
    test_dir = resolve_dataset_path(PRIMARY_TEST_DIR, FALLBACK_TEST_DIR)
    test_image_path = resolve_gradcam_image_path(test_dir)

    print(f"Using device: {device}")
    print(f"Training directory: {train_dir}")
    print(f"Testing directory: {test_dir}")
    print(f"GradCAM image: {test_image_path}")

    train_loader, val_loader, _, class_names, eval_transform, train_targets = create_dataloaders(
        train_dir,
        test_dir,
    )

    print(f"Classes: {class_names}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    model = build_model(device)
    set_backbone_trainable(model, trainable=False)

    class_weights = build_class_weights(train_targets, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE * 2.0,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE * 0.1)

    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_val_accuracy = -1.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, EPOCHS + 1):
        if epoch == FROZEN_BACKBONE_EPOCHS + 1:
            set_backbone_trainable(model, trainable=True)
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
            device,
            epoch,
        )
        val_loss, val_accuracy = validate_one_epoch(
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
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), str(BEST_MODEL_PATH))
            print(
                f"Saved best validation model to {BEST_MODEL_PATH} "
                f"with validation accuracy {val_accuracy:.4f}"
            )

    torch.save(model.state_dict(), str(FINAL_MODEL_PATH))
    print(f"Saved final trained model to {FINAL_MODEL_PATH}")

    model.load_state_dict(best_weights)
    predicted_class, confidence = generate_gradcam_visualization(
        model,
        test_image_path,
        eval_transform,
        class_names,
        device,
    )

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Predicted tumor class: {predicted_class}")
    print(f"Prediction confidence: {confidence:.4f}")
    print(f"Saved GradCAM visualization to {GRADCAM_OUTPUT_PATH}")


if __name__ == "__main__":
    main()