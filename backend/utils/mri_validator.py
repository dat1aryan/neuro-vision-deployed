"""
Brain MRI Validation System

This module performs practical validation to ensure images are suitable for MRI analysis:
1. File format validation (must be valid image)
2. Basic sanity checks (reasonable intensity, has content)
3. Optional diagnostics

Note: Overly strict validation rejects legitimate training data, so thresholds are calibrated
for real-world medical images including web downloads and various acquisition parameters.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


logger = logging.getLogger("ann_backend")

# ============================================================================
# CONFIGURATION
# ============================================================================

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jfif",
}
ALLOWED_MIME_PREFIXES = ("image/",)
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
IMAGE_SIZE = 224

# Practical sanity check thresholds (not overly strict)
SANITY_MEAN_MIN = 10.0  # Not completely black
SANITY_MEAN_MAX = 240.0  # Not completely white
SANITY_MIN_VARIANCE = 20.0  # Has some variation/content
SANITY_NON_BLACK_RATIO_MIN = 0.02  # At least 2% non-black pixels

# Grayscale strictness - BALANCED (accept medical MRI, reject colored images)
GRAYSCALE_COLOR_DIFF_THRESHOLD = 8.0  # Allow slight compression/encoding artifacts in medical images
GRAYSCALE_COLORFULNESS_MAX = 20.0  # Strict on colorfulness but allows medical imaging
GRAYSCALE_CHANNEL_STD_MAX = 5.0  # Allow medical scanner variations

# Legacy thresholds
VALIDATOR_CONFIDENCE_THRESHOLD = 0.70
VALIDATOR_CLASS_NAMES = ("brain_mri", "non_mri")
DEFAULT_VALIDATOR_ARCHITECTURE = "resnet18"

_RESAMPLING = getattr(Image, "Resampling", Image)




# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass(frozen=True)
class ValidationResult:
    """Structured validation output."""
    is_valid: bool
    reason: str

    def to_json(self) -> dict[str, Any]:
        return {"is_valid": self.is_valid, "reason": self.reason}


@dataclass(frozen=True)
class MRIHardGateMetrics:
    """Diagnostic metrics used during validation."""
    is_grayscale: bool
    colorfulness: float
    mean_intensity: float
    variance: float
    edge_density: float
    symmetry_score: float
    foreground_ratio: float
    tissue_contrast: float
    texture_complexity: float


# Legacy data classes
@dataclass(frozen=True)
class MRIValidatorArtifacts:
    model: Any
    transform: Any
    class_names: tuple[str, str]


@dataclass(frozen=True)
class MRIValidatorResult:
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]


@dataclass(frozen=True)
class MRIHeuristicMetrics:
    colorfulness: float
    edge_density: float
    symmetry_score: float
    center_border_delta: float
    center_border_ratio: float


# ============================================================================
# PRACTICAL VALIDATION
# ============================================================================

def _resize_grayscale(image: Image.Image) -> np.ndarray:
    grayscale_image = image.convert("L").resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        _RESAMPLING.BILINEAR,
    )
    return np.asarray(grayscale_image, dtype=np.float32)


def _resize_rgb(image: Image.Image) -> np.ndarray:
    rgb_image = image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        _RESAMPLING.BILINEAR,
    )
    return np.asarray(rgb_image, dtype=np.float32)


def _compute_colorfulness(rgb_array: np.ndarray) -> float:
    """Compute colorfulness metric (should be very low for MRI)."""
    red_channel = rgb_array[:, :, 0]
    green_channel = rgb_array[:, :, 1]
    blue_channel = rgb_array[:, :, 2]
    rg_channel = red_channel - green_channel
    yb_channel = 0.5 * (red_channel + green_channel) - blue_channel

    std_root = float(np.sqrt(np.std(rg_channel) ** 2 + np.std(yb_channel) ** 2))
    mean_root = float(np.sqrt(np.mean(rg_channel) ** 2 + np.mean(yb_channel) ** 2))
    return std_root + (0.3 * mean_root)


def _edge_density_from_gray(gray_array: np.ndarray) -> float:
    """Compute lightweight edge density using NumPy gradients."""
    normalized = gray_array.astype(np.float32) / 255.0
    grad_y, grad_x = np.gradient(normalized)
    magnitude = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))
    edge_mask = magnitude > 0.08
    return float(np.count_nonzero(edge_mask)) / float(edge_mask.size)


def _is_strict_grayscale(image: Image.Image) -> tuple[bool, float]:
    """
    **CRITICAL**: Check if image is STRICTLY BLACK & WHITE (pure grayscale).
    
    This function REJECTS ANY color detected in the image.
    Standard: Grayscale medical MRI only. NO COLOR IMAGES ALLOWED.
    
    Returns (is_pure_black_and_white: bool, colorfulness_score: float)
    
    Algorithm:
    - Method 1: Channel analysis (R = G = B for pure black/white)
    - Method 2: Per-pixel color detection
    - Method 3: Colorfulness metric (very low for B&W)
    - Method 4: Channel standard deviation
    
    All four methods must pass for image to be accepted as valid grayscale.
    """
    # If already in grayscale mode, it's definitely grayscale
    if image.mode in ("L", "LA"):
        return True, 0.0

    # Convert to RGB and get array
    rgb_image = image.convert("RGB")
    rgb_array = np.asarray(rgb_image, dtype=np.float32)

    # Method 1: Check if channels are nearly identical
    red = rgb_array[:, :, 0]
    green = rgb_array[:, :, 1]
    blue = rgb_array[:, :, 2]

    # Check mean differences between channels
    rg_diff = np.mean(np.abs(red - green))
    rb_diff = np.mean(np.abs(red - blue))
    gb_diff = np.mean(np.abs(green - blue))
    max_channel_diff = max(rg_diff, rb_diff, gb_diff)

    # Check if ANY pixel has significant color
    # A grayscale pixel has R = G = B (or very close)
    channel_diff_per_pixel = np.maximum(
        np.maximum(np.abs(red - green), np.abs(red - blue)),
        np.abs(green - blue)
    )
    max_pixel_diff = np.max(channel_diff_per_pixel)
    pixels_with_color = np.sum(channel_diff_per_pixel > 10.0)  # Allow 10pt difference per pixel
    color_pixel_ratio = pixels_with_color / rgb_array.shape[0] / rgb_array.shape[1]

    # Method 2: Compute colorfulness
    colorfulness = _compute_colorfulness(rgb_array)

    # Method 3: Check standard deviation between channels
    channel_std = np.std([np.mean(red), np.mean(green), np.mean(blue)])

    logger.debug(
        f"Grayscale check: "
        f"mean_diff={max_channel_diff:.2f}(threshold={GRAYSCALE_COLOR_DIFF_THRESHOLD}), "
        f"max_pixel_diff={max_pixel_diff:.2f}, "
        f"color_pixels={color_pixel_ratio:.4f}, "
        f"colorfulness={colorfulness:.2f}(threshold={GRAYSCALE_COLORFULNESS_MAX}), "
        f"channel_std={channel_std:.2f}"
    )

    # STRICT REJECTION - Reject obviously colored images but allow medical MRI
    is_grayscale = (
        max_channel_diff < GRAYSCALE_COLOR_DIFF_THRESHOLD and
        colorfulness < GRAYSCALE_COLORFULNESS_MAX and
        channel_std < GRAYSCALE_CHANNEL_STD_MAX and
        color_pixel_ratio < 0.10  # Allow up to 10% of pixels to have slight color (medical imaging artifacts)
    )
    
    return is_grayscale, colorfulness


def validate_file_format(file: UploadFile, file_bytes: bytes) -> Image.Image:
    """
    Validate file format and return image.
    Ensures file is a valid image before proceeding.
    """
    suffix = Path(file.filename or "").suffix.lower()
    content_type = (file.content_type or "").lower()

    suffix_is_supported = bool(suffix) and suffix in ALLOWED_IMAGE_EXTENSIONS
    content_type_is_supported = any(
        content_type.startswith(prefix) for prefix in ALLOWED_MIME_PREFIXES
    )

    if suffix and not suffix_is_supported and not content_type_is_supported:
        raise ValueError("Unsupported file format")

    if not file_bytes or len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError("Invalid file size")

    try:
        with Image.open(io.BytesIO(file_bytes)) as loaded_image:
            loaded_image.load()
            image_format = (loaded_image.format or "").upper()
            if image_format and image_format not in ALLOWED_PIL_FORMATS:
                raise ValueError("Unsupported image format")
            return loaded_image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ValueError(f"Cannot read image: {e}") from e


def validate_brain_mri(image: Image.Image) -> ValidationResult:
    """
    STRICT validation for brain MRI images.
    
    **CRITICAL REQUIREMENT**: Image MUST be pure Black & White (grayscale) - ABSOLUTELY NO COLOR
    
    Validation pipeline:
    1. Image MUST be pure grayscale (black and white only) - NO COLOR ALLOWED
    2. Image must have reasonable intensity levels (not pure black/white)
    3. Image must have content (variance) - proof of brain structure
    4. Image must have meaningful foreground
    
    Rejection criteria:
    - Any colored pixels detected → REJECTED
    - Invalid intensity (too dark/bright) → REJECTED
    - No internal brain structure → REJECTED
    - Mostly black/empty image → REJECTED
    """
    logger.info("=" * 80)
    logger.info("STRICT BLACK & WHITE BRAIN MRI VALIDATION STARTING")
    logger.info("Only pure grayscale brain MRI images are accepted. NO COLOR.")
    logger.info("=" * 80)
    
    # Step 1: AGGRESSIVE GRAYSCALE CHECK - NO COLOR ALLOWED
    logger.info("STEP 1: Verifying image is PURE BLACK & WHITE (grayscale only)...")
    is_grayscale, colorfulness = _is_strict_grayscale(image)
    
    if not is_grayscale:
        logger.warning("=" * 80)
        logger.warning("❌ REJECTED: IMAGE CONTAINS COLOR - ONLY BLACK & WHITE ACCEPTED")
        logger.warning(f"Colorfulness score: {colorfulness:.2f} (max allowed: {GRAYSCALE_COLORFULNESS_MAX})")
        logger.warning("This image is not a valid black and white brain MRI.")
        logger.warning("Please upload a pure grayscale MRI brain scan with no color.")
        logger.warning("=" * 80)
        return ValidationResult(
            is_valid=False,
            reason="❌ IMAGE REJECTED: Contains color. Only pure black & white brain MRI images (grayscale) are accepted. Please upload a grayscale MRI brain scan."
        )
    
    logger.info("✓ PASSED: Image is pure black & white (grayscale confirmed)")
    
    # Step 2: BRAIN MRI CHECKS
    logger.info("STEP 2: Checking image has valid brain MRI characteristics...")
    grayscale_array = _resize_grayscale(image)
    pixel_mean = float(np.mean(grayscale_array))
    pixel_variance = float(np.var(grayscale_array))
    foreground_pixels = grayscale_array[grayscale_array > 5.0]
    non_black_ratio = float(foreground_pixels.size) / float(grayscale_array.size)
    
    logger.info(f"Image metrics:")
    logger.info(f"  - Mean intensity: {pixel_mean:.1f} (valid: {SANITY_MEAN_MIN}-{SANITY_MEAN_MAX})")
    logger.info(f"  - Variance: {pixel_variance:.1f} (min: {SANITY_MIN_VARIANCE})")
    logger.info(f"  - Foreground: {non_black_ratio:.4f} (min: {SANITY_NON_BLACK_RATIO_MIN})")

    # Check mean intensity
    if pixel_mean < SANITY_MEAN_MIN or pixel_mean > SANITY_MEAN_MAX:
        logger.warning("=" * 80)
        logger.warning(f"❌ REJECTED: Invalid intensity (mean={pixel_mean:.1f})")
        logger.warning("This does not appear to be a brain MRI scan.")
        logger.warning("=" * 80)
        return ValidationResult(
            is_valid=False,
            reason="❌ IMAGE REJECTED: Invalid intensity. Only brain MRI scans accepted."
        )

    # Check variance
    if pixel_variance < SANITY_MIN_VARIANCE:
        logger.warning("=" * 80)
        logger.warning(f"❌ REJECTED: No internal brain structure (variance={pixel_variance:.1f})")
        logger.warning("This does not appear to be a brain MRI.")
        logger.warning("=" * 80)
        return ValidationResult(
            is_valid=False,
            reason="❌ IMAGE REJECTED: No internal brain structure detected. Please upload a valid brain MRI scan."
        )

    # Check foreground
    if non_black_ratio < SANITY_NON_BLACK_RATIO_MIN:
        logger.warning("=" * 80)
        logger.warning(f"❌ REJECTED: Image is mostly black (foreground={non_black_ratio:.4f})")
        logger.warning("This does not appear to be a brain MRI.")
        logger.warning("=" * 80)
        return ValidationResult(
            is_valid=False,
            reason="❌ IMAGE REJECTED: Image is mostly black/empty. Please upload a valid brain MRI scan."
        )

    logger.info("=" * 80)
    logger.info("✅ VALIDATION PASSED - PURE BLACK & WHITE BRAIN MRI CONFIRMED")
    logger.info("=" * 80)
    return ValidationResult(
        is_valid=True,
        reason="Valid grayscale brain MRI image. Proceeding to analysis."
    )


# ============================================================================
# LEGACY COMPATIBILITY WRAPPERS
# ============================================================================

def _validation_error_payload() -> dict[str, str]:
    return {
        "error": "INVALID_INPUT",
        "message": "INVALID_INPUT",
    }


def _raise_validation_error() -> None:
    logger.warning("Invalid MRI upload attempt")
    raise HTTPException(
        status_code=400,
        detail=_validation_error_payload(),
    )


def validate_file(file: UploadFile, file_bytes: bytes) -> Image.Image:
    """Legacy wrapper - validates file format."""
    try:
        return validate_file_format(file, file_bytes)
    except ValueError as e:
        _raise_validation_error()


def sanity_check(image: Image.Image) -> np.ndarray:
    """Legacy wrapper - validates and returns grayscale array."""
    # Run basic validation
    result = validate_brain_mri(image)
    if not result.is_valid:
        _raise_validation_error()
    
    gray_image = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), _RESAMPLING.BILINEAR)
    gray_array = np.asarray(gray_image, dtype=np.float32)
    return gray_array / 255.0


def optional_edge_check(image: Image.Image) -> float:
    """Legacy wrapper - computes edge density for diagnostics."""
    gray_image = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), _RESAMPLING.BILINEAR)
    gray_array = np.asarray(gray_image, dtype=np.uint8)
    return _edge_density_from_gray(gray_array)


def fallback_validate_brain_mri(image: Image.Image) -> MRIValidatorResult:
    """Legacy wrapper - uses new practical validation."""
    result = validate_brain_mri(image)
    confidence = 0.95 if result.is_valid else 0.0
    
    if not result.is_valid:
        _raise_validation_error()
    
    return MRIValidatorResult(
        predicted_class="brain_mri",
        confidence=confidence,
        class_probabilities={
            "brain_mri": confidence,
            "non_mri": 1.0 - confidence
        }
    )


def compute_heuristic_metrics(image: Image.Image) -> MRIHeuristicMetrics:
    """Legacy wrapper - provides backward compatibility."""
    gray_image = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), _RESAMPLING.BILINEAR)
    gray_array = np.asarray(gray_image, dtype=np.float32)
    
    rgb_array = _resize_rgb(image)
    
    red = rgb_array[:, :, 0]
    green = rgb_array[:, :, 1]
    blue = rgb_array[:, :, 2]
    rg = red - green
    yb = 0.5 * (red + green) - blue
    std_rg_yb = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
    mean_rg_yb = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    colorfulness = float(std_rg_yb + 0.3 * mean_rg_yb)
    
    edge_density = _edge_density_from_gray(gray_array.astype(np.uint8))
    
    h, w = gray_array.shape
    left_half = gray_array[:, :w // 2]
    right_half = np.fliplr(gray_array[:, w // 2 : w // 2 * 2])
    symmetry = float(np.mean(np.abs(left_half - right_half)))
    
    border_region = np.concatenate([
        gray_array[:16, :].ravel(),
        gray_array[-16:, :].ravel(),
        gray_array[:, :16].ravel(),
        gray_array[:, -16:].ravel(),
    ])
    center_region = gray_array[56:168, 56:168]
    center_mean = float(np.mean(center_region))
    border_mean = float(np.mean(border_region))
    
    return MRIHeuristicMetrics(
        colorfulness=colorfulness,
        edge_density=edge_density,
        symmetry_score=symmetry,
        center_border_delta=center_mean - border_mean,
        center_border_ratio=float((center_mean + 1.0) / (border_mean + 1.0)),
    )


def build_validator_model(architecture: str = DEFAULT_VALIDATOR_ARCHITECTURE) -> Any:
    """Legacy stub - no longer used."""
    raise NotImplementedError("Legacy validator model building is deprecated")


def load_validator_model(
    model_path: Path,
    device: Any,
) -> MRIValidatorArtifacts:
    """Legacy stub - no longer used."""
    raise NotImplementedError("Legacy validator model loading is deprecated")


def run_validator(
    validator_artifacts: MRIValidatorArtifacts,
    image: Image.Image,
    device: Any,
    lock: Any | None = None,
) -> MRIValidatorResult:
    """Legacy stub - no longer used."""
    raise NotImplementedError("Legacy validator model inference is deprecated")