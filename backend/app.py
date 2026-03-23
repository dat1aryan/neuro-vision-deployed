"""
Install dependencies:
pip install fastapi uvicorn torch torchvision pillow numpy pandas scikit-learn joblib grad-cam opencv-python
pip install python-multipart

Run server:
uvicorn backend.app:app --reload
"""

from __future__ import annotations

import datetime
import html
import io
import json
import logging
import os
import threading
import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import cv2
from gradio_client import Client, handle_file
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn
from sklearn.pipeline import Pipeline
from torchvision import models, transforms
from pydantic import BaseModel
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend.utils.mri_validator import (
    optional_edge_check,
    validate_brain_mri,
    validate_file_format,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ann_backend")

PLATFORM_NAME = "Neuro Vision"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COGNITIVE_MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "cognitive_risk_model.pkl"
COGNITIVE_FEATURES_PATH = PROJECT_ROOT / "backend" / "models" / "cognitive_features.pkl"
PRIMARY_COGNITIVE_DATA_PATH = PROJECT_ROOT / "Data" / "alzheimers_disease_data.csv"
FALLBACK_COGNITIVE_DATA_PATH = PROJECT_ROOT / "Data" / "Dataset" / "alzheimers_disease_data.csv"
MRI_GRADCAM_MODEL_PATH = PROJECT_ROOT / "ml_service" / "models" / "mri_model_best.pth"
ML_SERVICE_API_URL = os.getenv(
    "ML_SERVICE_API_URL",
    "https://dat1aryan-neuro-vision-ml.hf.space",
).strip()
ML_SERVICE_TIMEOUT_SECONDS = float(os.getenv("ML_SERVICE_TIMEOUT_SECONDS", "45"))
COGNITIVE_TEST_HISTORY_LIMIT = 200
COGNITIVE_TEST_HISTORY: list[dict[str, Any]] = []
COGNITIVE_TEST_HISTORY_LOCK = threading.Lock()
MRI_HF_CLIENT = Client("dat1aryan/neuro-vision-ml")
MRI_GRADCAM_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MRI_GRADCAM_MODEL: nn.Module | None = None
MRI_GRADCAM_LOCK = threading.Lock()
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MRI_EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

PRIMARY_MRI_TRAIN_DIR = PROJECT_ROOT / "Data" / "Training"
PRIMARY_MRI_TEST_DIR = PROJECT_ROOT / "Data" / "Testing"
FALLBACK_MRI_TRAIN_DIR = PROJECT_ROOT / "Data" / "Dataset" / "Training"
FALLBACK_MRI_TEST_DIR = PROJECT_ROOT / "Data" / "Dataset" / "Testing"

MRI_TEMPERATURE = 1.0
MRI_CONFIDENCE_HIGH_THRESHOLD = 0.75
MRI_CONFIDENCE_MEDIUM_THRESHOLD = 0.50
MRI_UNCERTAINTY_LOW_THRESHOLD = 0.22
MRI_UNCERTAINTY_MEDIUM_THRESHOLD = 0.45

FUSION_MRI_WEIGHT = 0.60
FUSION_COGNITIVE_WEIGHT = 0.40

COGNITIVE_TARGET_ALIASES = ["cognitive_risk", "Diagnosis"]
COGNITIVE_UI_MODEL_FEATURES = ["Age", "EducationLevel", "MMSE", "FunctionalAssessment"]
COGNITIVE_AGE_MIN = 60.0
COGNITIVE_AGE_MAX = 90.0
COGNITIVE_EDUCATION_LEVEL_MAX = 3.0
COGNITIVE_MMSE_MAX = 30.0
COGNITIVE_FUNCTIONAL_MAX = 10.0
COGNITIVE_UI_MEMORY_MAX = 25.0
COGNITIVE_UI_COGNITIVE_MAX = 30.0
COGNITIVE_UI_EDUCATION_MAX_YEARS = 25.0
COGNITIVE_MODULE_MAX_SCORES = {
    "orientation": 3.0,
    "digit_forward": 2.0,
    "digit_backward": 2.0,
    "executive_reasoning": 2.0,
    "reaction": 2.0,
    "visual_pattern": 3.0,
    "verbal_fluency": 2.0,
    "category": 2.0,
    "stroop": 2.0,
    "symbol_digit": 2.0,
    "mental_arithmetic": 2.0,
    "spatial_rotation": 2.0,
    "delayed_recall": 6.0,
}
CLINICAL_DISCLAIMER = (
    "AI outputs are decision-support signals only and must be interpreted by a qualified clinician. "
    "This system is not a standalone diagnostic tool."
)

COGNITIVE_WORD_BANK = [
    "apple",
    "river",
    "mountain",
    "chair",
    "clock",
    "garden",
    "paper",
    "bridge",
    "pencil",
    "window",
    "camera",
    "flower",
    "ocean",
    "forest",
    "mirror",
    "lantern",
    "planet",
    "desert",
    "guitar",
    "shadow",
]

COGNITIVE_REASONING_QUESTIONS = [
    {
        "id": "reasoning_1",
        "prompt": "2, 4, 8, 16, ?",
        "options": ["20", "24", "32", "18"],
        "answer": "32",
    },
    {
        "id": "reasoning_2",
        "prompt": "3, 6, 12, 24, ?",
        "options": ["30", "36", "48", "54"],
        "answer": "48",
    },
    {
        "id": "reasoning_3",
        "prompt": "1, 1, 2, 3, 5, ?",
        "options": ["6", "7", "8", "9"],
        "answer": "8",
    },
    {
        "id": "reasoning_4",
        "prompt": "5, 10, 20, 40, ?",
        "options": ["60", "70", "80", "90"],
        "answer": "80",
    },
]

COGNITIVE_CATEGORY_QUESTIONS = [
    {
        "id": "category_1",
        "prompt": "Which item does not belong?",
        "options": ["Apple", "Banana", "Car", "Orange"],
        "answer": "Car",
    },
    {
        "id": "category_2",
        "prompt": "Which item does not belong?",
        "options": ["Triangle", "Square", "Circle", "Carrot"],
        "answer": "Carrot",
    },
    {
        "id": "category_3",
        "prompt": "Which item does not belong?",
        "options": ["Dog", "Cat", "Eagle", "Spoon"],
        "answer": "Spoon",
    },
    {
        "id": "category_4",
        "prompt": "Which item does not belong?",
        "options": ["Winter", "Summer", "Monday", "Spring"],
        "answer": "Monday",
    },
]

COGNITIVE_SPATIAL_ROTATION_QUESTIONS = [
    {
        "id": "spatial_1",
        "prompt": "Select the option that matches the target shape rotated 90 degrees clockwise.",
        "rotation_degrees": 90,
        "rotation_direction": "clockwise",
        "target": [[1, 0, 0], [1, 1, 1], [0, 0, 1]],
        "options": [
            {"id": "A", "grid": [[0, 1, 1], [0, 1, 0], [1, 1, 0]]},
            {"id": "B", "grid": [[1, 1, 0], [0, 1, 0], [0, 1, 1]]},
            {"id": "C", "grid": [[1, 0, 0], [1, 1, 1], [0, 0, 1]]},
            {"id": "D", "grid": [[0, 0, 1], [1, 1, 1], [1, 0, 0]]},
        ],
        "answer": "A",
    },
    {
        "id": "spatial_2",
        "prompt": "Select the option that matches the target shape rotated 180 degrees clockwise.",
        "rotation_degrees": 180,
        "rotation_direction": "clockwise",
        "target": [[1, 0, 0], [1, 1, 0], [0, 1, 1]],
        "options": [
            {"id": "A", "grid": [[0, 0, 1], [1, 1, 0], [1, 0, 0]]},
            {"id": "B", "grid": [[1, 1, 0], [0, 1, 1], [0, 0, 1]]},
            {"id": "C", "grid": [[1, 0, 0], [1, 1, 1], [0, 0, 1]]},
            {"id": "D", "grid": [[0, 1, 0], [1, 1, 0], [1, 0, 1]]},
        ],
        "answer": "B",
    },
    {
        "id": "spatial_3",
        "prompt": "Select the option that matches the target shape rotated 180 degrees anticlockwise.",
        "rotation_degrees": 180,
        "rotation_direction": "anticlockwise",
        "target": [[0, 1, 0], [1, 1, 0], [1, 0, 0]],
        "options": [
            {"id": "A", "grid": [[0, 1, 1], [0, 1, 0], [0, 0, 1]]},
            {"id": "B", "grid": [[1, 0, 0], [1, 1, 0], [0, 1, 0]]},
            {"id": "C", "grid": [[0, 0, 1], [0, 1, 1], [0, 1, 0]]},
            {"id": "D", "grid": [[0, 1, 0], [1, 1, 0], [1, 0, 0]]},
        ],
        "answer": "C",
    },
    {
        "id": "spatial_4",
        "prompt": "Select the option that matches the target shape rotated 90 degrees anticlockwise.",
        "rotation_degrees": 90,
        "rotation_direction": "anticlockwise",
        "target": [[1, 0, 0], [1, 1, 0], [0, 1, 1]],
        "options": [
            {"id": "A", "grid": [[0, 1, 1], [1, 1, 0], [1, 0, 0]]},
            {"id": "B", "grid": [[0, 0, 1], [0, 1, 1], [1, 1, 0]]},
            {"id": "C", "grid": [[1, 0, 0], [1, 1, 0], [0, 1, 1]]},
            {"id": "D", "grid": [[1, 1, 0], [0, 1, 1], [0, 0, 1]]},
        ],
        "answer": "B",
    },
    {
        "id": "spatial_5",
        "prompt": "Select the option that matches the target shape rotated 90 degrees clockwise.",
        "rotation_degrees": 90,
        "rotation_direction": "clockwise",
        "target": [[0, 1, 0], [1, 1, 0], [1, 0, 0]],
        "options": [
            {"id": "A", "grid": [[0, 1, 0], [1, 1, 0], [1, 0, 0]]},
            {"id": "B", "grid": [[1, 1, 0], [0, 1, 1], [0, 0, 0]]},
            {"id": "C", "grid": [[0, 0, 0], [1, 1, 0], [0, 1, 1]]},
            {"id": "D", "grid": [[0, 0, 1], [0, 1, 1], [0, 1, 0]]},
        ],
        "answer": "B",
    },
    {
        "id": "spatial_6",
        "prompt": "Select the option that matches the target shape rotated 90 degrees anticlockwise.",
        "rotation_degrees": 90,
        "rotation_direction": "anticlockwise",
        "target": [[0, 1, 1], [1, 1, 0], [0, 0, 0]],
        "options": [
            {"id": "A", "grid": [[0, 1, 0], [0, 1, 1], [0, 0, 1]]},
            {"id": "B", "grid": [[0, 1, 1], [1, 1, 0], [0, 0, 0]]},
            {"id": "C", "grid": [[0, 0, 0], [0, 1, 1], [1, 1, 0]]},
            {"id": "D", "grid": [[1, 0, 0], [1, 1, 0], [0, 1, 0]]},
        ],
        "answer": "D",
    },
]

COGNITIVE_VISUAL_PATTERN_BANK = [
    {"id": "pattern_1", "cells": [0, 4, 8]},
    {"id": "pattern_2", "cells": [1, 3, 5]},
    {"id": "pattern_3", "cells": [2, 4, 6]},
    {"id": "pattern_4", "cells": [0, 1, 4]},
    {"id": "pattern_5", "cells": [3, 4, 5]},
    {"id": "pattern_6", "cells": [1, 4, 7]},
]

COGNITIVE_STROOP_TRIALS = [
    {"id": "stroop_1", "word": "RED", "display_color": "blue", "options": ["red", "blue", "green", "yellow"]},
    {"id": "stroop_2", "word": "GREEN", "display_color": "red", "options": ["red", "blue", "green", "yellow"]},
    {"id": "stroop_3", "word": "BLUE", "display_color": "yellow", "options": ["red", "blue", "green", "yellow"]},
    {"id": "stroop_4", "word": "YELLOW", "display_color": "green", "options": ["red", "blue", "green", "yellow"]},
]

COGNITIVE_SYMBOL_DIGIT_TRIALS = [
    {
        "id": "symbol_1",
        "mapping": {"triangle": "1", "circle": "2", "square": "3", "diamond": "4"},
        "sequence": ["triangle", "circle", "square", "circle"],
        "answer": ["1", "2", "3", "2"],
    },
    {
        "id": "symbol_2",
        "mapping": {"triangle": "1", "circle": "2", "square": "3", "diamond": "4"},
        "sequence": ["diamond", "square", "triangle", "circle"],
        "answer": ["4", "3", "1", "2"],
    },
    {
        "id": "symbol_3",
        "mapping": {"triangle": "1", "circle": "2", "square": "3", "diamond": "4"},
        "sequence": ["circle", "diamond", "square", "triangle"],
        "answer": ["2", "4", "3", "1"],
    },
]

COGNITIVE_MENTAL_ARITHMETIC_BANK = [
    {"id": "arith_1", "prompt": "93 - 7", "answer": 86},
    {"id": "arith_2", "prompt": "86 - 7", "answer": 79},
    {"id": "arith_3", "prompt": "79 - 7", "answer": 72},
    {"id": "arith_4", "prompt": "72 - 7", "answer": 65},
    {"id": "arith_5", "prompt": "65 - 7", "answer": 58},
]

COGNITIVE_ANIMAL_LEXICON = [
    "ant",
    "bear",
    "cat",
    "cow",
    "deer",
    "dog",
    "duck",
    "eagle",
    "elephant",
    "fox",
    "frog",
    "goat",
    "horse",
    "kangaroo",
    "lion",
    "monkey",
    "owl",
    "panda",
    "rabbit",
    "seal",
    "shark",
    "sheep",
    "snake",
    "sparrow",
    "tiger",
    "turtle",
    "whale",
    "wolf",
    "zebra",
]


@dataclass
class ANNArtifacts:
    cognitive_model: Pipeline
    cognitive_saved_schema: list[str]
    cognitive_input_features: list[str]
    cognitive_defaults: dict[str, Any]
    cognitive_lock: threading.Lock


def load_saved_cognitive_schema() -> list[str]:
    if not COGNITIVE_FEATURES_PATH.is_file():
        raise FileNotFoundError(
            f"Cognitive feature schema file not found at {COGNITIVE_FEATURES_PATH}"
        )

    saved_schema = joblib.load(COGNITIVE_FEATURES_PATH)

    if isinstance(saved_schema, dict):
        if "features" in saved_schema:
            saved_schema = saved_schema["features"]
        elif "transformed_features" in saved_schema:
            saved_schema = saved_schema["transformed_features"]

    if not isinstance(saved_schema, list) or not all(
        isinstance(item, str) for item in saved_schema
    ):
        raise RuntimeError(
            "Saved cognitive feature schema must be a list of feature names."
        )

    return saved_schema


def load_cognitive_model() -> Pipeline:
    if not COGNITIVE_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Cognitive model file not found at {COGNITIVE_MODEL_PATH}"
        )

    logger.info("%s loading cognitive model from %s", PLATFORM_NAME, COGNITIVE_MODEL_PATH)
    model = joblib.load(COGNITIVE_MODEL_PATH)
    if not isinstance(model, Pipeline):
        raise RuntimeError("Cognitive model must be a scikit-learn Pipeline.")

    return model


def resolve_cognitive_input_features(
    cognitive_model: Pipeline,
    saved_schema: list[str],
) -> list[str]:
    raw_features = getattr(cognitive_model, "feature_names_in_", None)
    if raw_features is None and hasattr(cognitive_model, "named_steps"):
        preprocessor = cognitive_model.named_steps.get("preprocessor")
        raw_features = getattr(preprocessor, "feature_names_in_", None)

    if raw_features is None:
        raise RuntimeError(
            "Could not resolve raw cognitive input features from the trained pipeline."
        )

    input_features = [str(feature) for feature in raw_features]
    logger.info(
        "%s loaded cognitive schema with %d transformed features and %d raw input features",
        PLATFORM_NAME,
        len(saved_schema),
        len(input_features),
    )
    return input_features


def resolve_cognitive_data_path() -> Path | None:
    if PRIMARY_COGNITIVE_DATA_PATH.is_file():
        return PRIMARY_COGNITIVE_DATA_PATH
    if FALLBACK_COGNITIVE_DATA_PATH.is_file():
        return FALLBACK_COGNITIVE_DATA_PATH
    return None


def build_cognitive_defaults(input_features: list[str]) -> dict[str, Any]:
    defaults: dict[str, Any] = {feature: 0.0 for feature in input_features}
    data_path = resolve_cognitive_data_path()

    if data_path is None:
        logger.warning(
            "%s could not find the cognitive CSV for default feature values; using zero defaults",
            PLATFORM_NAME,
        )
        return defaults

    dataframe = pd.read_csv(data_path)

    for feature in input_features:
        if feature not in dataframe.columns:
            continue

        series = dataframe[feature].dropna()
        if series.empty:
            continue

        if pd.api.types.is_numeric_dtype(series):
            defaults[feature] = float(series.median())
        else:
            mode = series.mode(dropna=True)
            defaults[feature] = str(mode.iloc[0]) if not mode.empty else ""

    logger.info("%s prepared defaults for %d cognitive features", PLATFORM_NAME, len(defaults))
    return defaults


def load_ann_artifacts() -> ANNArtifacts:
    cognitive_model = load_cognitive_model()
    saved_schema = load_saved_cognitive_schema()
    input_features = resolve_cognitive_input_features(cognitive_model, saved_schema)
    feature_defaults = build_cognitive_defaults(input_features)

    return ANNArtifacts(
        cognitive_model=cognitive_model,
        cognitive_saved_schema=saved_schema,
        cognitive_input_features=input_features,
        cognitive_defaults=feature_defaults,
        cognitive_lock=threading.Lock(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("%s starting up", PLATFORM_NAME)
    app.state.ann = load_ann_artifacts()
    logger.info("%s startup complete", PLATFORM_NAME)
    try:
        yield
    finally:
        logger.info("%s shutting down", PLATFORM_NAME)


app = FastAPI(title=PLATFORM_NAME, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def normalize_key(name: str) -> str:
    return "".join(character.lower() for character in name if character.isalnum())


def build_cognitive_alias_map(input_features: list[str]) -> dict[str, str]:
    alias_map = {normalize_key(feature): feature for feature in input_features}
    manual_aliases = {
        "education": "EducationLevel",
        "educationlevel": "EducationLevel",
        "memoryscore": "MMSE",
        "memorytestscore": "MMSE",
        "cognitivescore": "FunctionalAssessment",
        "functionalassessment": "FunctionalAssessment",
        "bloodpressurehigh": "SystolicBP",
        "bloodpressurelow": "DiastolicBP",
    }

    for alias, feature in manual_aliases.items():
        if feature in input_features:
            alias_map[alias] = feature

    return alias_map


def parse_clinical_json_payload(clinical_json: str) -> dict[str, Any]:
    try:
        payload = json.loads(clinical_json)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=400,
            detail="clinical_json must be a valid JSON object string",
        ) from error

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail="clinical_json must decode to a JSON object",
        )

    return payload


async def resolve_clinical_payload(
    request: Request,
    clinical_json: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    if clinical_json is not None and clinical_json.strip():
        payload.update(parse_clinical_json_payload(clinical_json))

    form = await request.form()
    for key, value in form.multi_items():
        if key in {"file", "clinical_json"}:
            continue

        if hasattr(value, "filename"):
            continue

        if isinstance(value, str):
            stripped_value = value.strip()
            if not stripped_value:
                continue
            payload[key] = stripped_value
            continue

        if value is not None:
            payload[key] = value

    if not payload:
        raise HTTPException(status_code=400, detail="Cognitive inputs are required")

    return payload


async def read_uploaded_image(file: UploadFile) -> tuple[Image.Image, bytes]:
    """
    HARD GATE IMAGE VALIDATION
    
    This function enforces validation rules:
    1. File must be a valid image
    2. Image must have reasonable content (not pure black/white)
    
    ONLY IF CONDITIONS ARE MET: proceed to analysis
    OTHERWISE: immediately reject and return error
    """
    image_bytes = await file.read()
    
    # Step 1: Validate file format
    try:
        image = validate_file_format(file, image_bytes)
    except ValueError as e:
        logger.warning("File format validation failed: %s", e)
        raise HTTPException(
            status_code=400,
            detail="Invalid image file format"
        )
    
    # Step 2: Run validation
    validation_result = validate_brain_mri(image)
    
    if not validation_result.is_valid:
        logger.warning("Image validation failed: %s", validation_result.reason)
        raise HTTPException(
            status_code=400,
            detail=validation_result.reason
        )
    
    # If validation passed, proceed
    logger.info("Image passed validation: %s", validation_result.reason)
    optional_edge_check(image)
    return image, image_bytes


def predict_mri_image(file_path: str) -> dict[str, Any]:
    try:
        result = MRI_HF_CLIENT.predict(
            image=handle_file(file_path),
            api_name="/predict",
        )
    except Exception as error:
        logger.exception("%s MRI prediction failed via HuggingFace", PLATFORM_NAME)
        raise HTTPException(status_code=500, detail=f"MRI prediction failed: {error}") from error

    if isinstance(result, list) and result:
        result = result[0]

    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="ML service returned invalid prediction result")

    prediction_value = result.get("prediction")
    confidence_value = result.get("confidence")

    if prediction_value is None or confidence_value is None:
        raise HTTPException(status_code=502, detail="ML service response missing prediction fields")

    try:
        confidence = float(confidence_value)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=502, detail="ML service returned invalid confidence value") from error

    prediction = str(prediction_value).strip().lower()
    tumor_probability = confidence if prediction != "notumor" else _clamp(1.0 - confidence, 0.0, 1.0)
    notumor_probability = confidence if prediction == "notumor" else _clamp(1.0 - confidence, 0.0, 1.0)

    if confidence >= MRI_CONFIDENCE_HIGH_THRESHOLD:
        confidence_level = "High"
    elif confidence >= MRI_CONFIDENCE_MEDIUM_THRESHOLD:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"

    uncertainty_score = _clamp(1.0 - confidence, 0.0, 1.0)
    class_probabilities = {class_name: 0.0 for class_name in CLASS_NAMES}
    class_probabilities[prediction] = round(confidence, 4)

    return {
        "prediction": prediction,
        "tumor_prediction": prediction,
        "confidence": confidence,
        "tumor_confidence": confidence,
        "raw_confidence": confidence,
        "mri_uncertainty": uncertainty_score,
        "uncertainty_score": uncertainty_score,
        "uncertainty_variance": 0.0,
        "uncertainty_entropy": 0.0,
        "confidence_level": confidence_level,
        "temperature": MRI_TEMPERATURE,
        "tumor_probability": tumor_probability,
        "notumor_probability": notumor_probability,
        "class_probabilities": class_probabilities,
    }


def predict_mri_image_from_image(image: Image.Image, filename_hint: str = "image.png") -> tuple[dict[str, Any], Image.Image]:
    """
    Predict MRI classification and return both prediction and the preprocessed image.
    
    Returns:
        (prediction_dict, prepared_image) - The MRI prediction and the cropped/prepared image used for inference
    """
    prepared_image = image.convert("RGB")
    suffix = Path(filename_hint or "image.png").suffix or ".png"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        prepared_image.save(temp_file, format="PNG")
        temp_path = temp_file.name

    try:
        prediction = predict_mri_image(temp_path)
        return prediction, prepared_image
    finally:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove temporary MRI file: %s", temp_path)


def _build_gradcam_model(uses_dropout_head: bool) -> nn.Module:
    if hasattr(models, "ResNet50_Weights"):
        model = models.resnet50(weights=None)
    else:
        model = models.resnet50(pretrained=False)

    if uses_dropout_head:
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.fc.in_features, len(CLASS_NAMES)),
        )
    else:
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    return model


def _get_gradcam_model() -> nn.Module:
    global MRI_GRADCAM_MODEL

    with MRI_GRADCAM_LOCK:
        if MRI_GRADCAM_MODEL is not None:
            return MRI_GRADCAM_MODEL

        if not MRI_GRADCAM_MODEL_PATH.is_file():
            raise FileNotFoundError(f"GradCAM MRI model not found at {MRI_GRADCAM_MODEL_PATH}")

        checkpoint = torch.load(MRI_GRADCAM_MODEL_PATH, map_location=MRI_GRADCAM_DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        uses_dropout_head = any(
            key.startswith("fc.1.") or ".fc.1." in key for key in state_dict.keys()
        )

        model = _build_gradcam_model(uses_dropout_head)
        model.load_state_dict(state_dict)
        model.to(MRI_GRADCAM_DEVICE)
        model.eval()

        MRI_GRADCAM_MODEL = model
        return MRI_GRADCAM_MODEL


def _prepare_gradcam_inputs(
    image_input: Image.Image | bytes | bytearray,
) -> tuple[torch.Tensor, np.ndarray]:
    if isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        image = Image.open(io.BytesIO(bytes(image_input))).convert("RGB")
    resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    rgb_image = np.array(resized_image, dtype=np.float32) / 255.0
    input_tensor = MRI_EVAL_TRANSFORM(resized_image).unsqueeze(0).to(MRI_GRADCAM_DEVICE)
    return input_tensor, rgb_image


def _predict_with_gradcam_model(
    image_input: Image.Image | bytes | bytearray,
) -> dict[str, Any]:
    model = _get_gradcam_model()
    input_tensor, _ = _prepare_gradcam_inputs(image_input)

    with torch.inference_mode():
        logits = model(input_tensor)
        probability_tensor = torch.softmax(logits, dim=1).squeeze(0)

    probability_array = probability_tensor.detach().cpu().numpy().astype(np.float64)
    predicted_index = int(np.argmax(probability_array))
    prediction = CLASS_NAMES[predicted_index]
    confidence = float(probability_array[predicted_index])

    entropy = -float(np.sum(probability_array * np.log(np.clip(probability_array, 1e-8, 1.0))))
    normalized_entropy = entropy / float(np.log(max(2, len(CLASS_NAMES))))
    uncertainty_score = float(_clamp(normalized_entropy, 0.0, 1.0))

    if confidence >= MRI_CONFIDENCE_HIGH_THRESHOLD:
        confidence_level = "High"
    elif confidence >= MRI_CONFIDENCE_MEDIUM_THRESHOLD:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"

    tumor_probability = confidence if prediction != "notumor" else _clamp(1.0 - confidence, 0.0, 1.0)
    notumor_probability = confidence if prediction == "notumor" else _clamp(1.0 - confidence, 0.0, 1.0)
    class_probabilities = {
        class_name: round(float(probability_array[idx]), 4)
        for idx, class_name in enumerate(CLASS_NAMES)
    }

    return {
        "prediction": prediction,
        "tumor_prediction": prediction,
        "confidence": confidence,
        "tumor_confidence": confidence,
        "raw_confidence": confidence,
        "mri_uncertainty": uncertainty_score,
        "uncertainty_score": uncertainty_score,
        "uncertainty_variance": 0.0,
        "uncertainty_entropy": round(float(normalized_entropy), 4),
        "confidence_level": confidence_level,
        "temperature": MRI_TEMPERATURE,
        "tumor_probability": tumor_probability,
        "notumor_probability": notumor_probability,
        "class_probabilities": class_probabilities,
    }


def _resolve_gradcam_target_prediction(
    image_input: Image.Image,
    filename_hint: str,
) -> dict[str, Any]:
    local_prediction = _predict_with_gradcam_model(image_input)

    try:
        cloud_prediction, _ = predict_mri_image_from_image(image_input, filename_hint)
    except Exception:
        logger.warning(
            "%s cloud MRI insight unavailable for GradCAM; using local prediction",
            PLATFORM_NAME,
        )
        result = dict(local_prediction)
        result["prediction_source"] = "local_fallback"
        return result

    local_probs = local_prediction.get("class_probabilities", {})
    local_notumor = float(local_probs.get("notumor", 0.0))
    local_label = str(local_prediction.get("prediction") or "").lower()
    local_confidence = float(local_prediction.get("confidence") or 0.0)

    cloud_label = str(cloud_prediction.get("prediction") or "").lower()
    cloud_confidence = float(cloud_prediction.get("confidence") or 0.0)

    # Guard no-tumor false positives by requiring strong evidence for tumor targeting.
    if cloud_label == "notumor" and (cloud_confidence >= 0.58 or local_notumor >= 0.52):
        result = dict(local_prediction)
        result["prediction"] = "notumor"
        result["tumor_prediction"] = "notumor"
        result["confidence"] = max(cloud_confidence, local_notumor)
        result["tumor_confidence"] = result["confidence"]
        result["raw_confidence"] = result["confidence"]
        result["prediction_source"] = "cloud_local_consensus"
        return result

    if local_label == "notumor" and local_confidence >= 0.78 and cloud_confidence < 0.72:
        result = dict(local_prediction)
        result["prediction_source"] = "local_notumor_override"
        return result

    if cloud_label != "notumor" and cloud_confidence >= 0.72 and local_notumor < 0.60:
        result = dict(local_prediction)
        result["prediction"] = cloud_label
        result["tumor_prediction"] = cloud_label
        result["confidence"] = cloud_confidence
        result["tumor_confidence"] = cloud_confidence
        result["raw_confidence"] = cloud_confidence
        result["prediction_source"] = "cloud_primary"
        return result

    result = dict(local_prediction)
    result["prediction_source"] = "local_primary_with_cloud_hint"
    return result


def _build_brain_mask(rgb_image: np.ndarray) -> np.ndarray:
    gray_uint8 = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray_uint8, (5, 5), 0)
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(otsu_mask)
    if num_labels > 1:
        largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        brain_mask = (labels == largest_idx).astype(np.float32)
    else:
        brain_mask = (otsu_mask > 0).astype(np.float32)

    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    brain_mask = cv2.GaussianBlur(brain_mask, (0, 0), sigmaX=1.5, sigmaY=1.5)
    return np.clip(brain_mask, 0.0, 1.0)


def _compute_ensemble_cam(model: nn.Module, input_tensor: torch.Tensor, target_index: int) -> np.ndarray:
    targets = [ClassifierOutputTarget(target_index)]
    target_layers = [model.layer3[-1].conv3, model.layer4[-1].conv3]

    cam_variants: list[tuple[np.ndarray, float]] = []
    for cam_cls, variant_weight in ((GradCAMPlusPlus, 0.65), (GradCAM, 0.35)):
        with cam_cls(model=model, target_layers=target_layers) as cam_extractor:
            base_cam = cam_extractor(input_tensor=input_tensor, targets=targets)[0]

            # Test-time augmentation: horizontal flip CAM, then unflip and blend.
            flipped_input = torch.flip(input_tensor, dims=[3])
            flipped_cam = cam_extractor(input_tensor=flipped_input, targets=targets)[0]

        flipped_cam = np.flip(flipped_cam, axis=1)
        blended_cam = (0.6 * base_cam) + (0.4 * flipped_cam)
        cam_variants.append((blended_cam.astype(np.float32), variant_weight))

    cam_sum = np.zeros_like(cam_variants[0][0], dtype=np.float32)
    weight_sum = 0.0
    for cam_map, weight in cam_variants:
        cam_sum += cam_map * float(weight)
        weight_sum += float(weight)

    merged_cam = cam_sum / max(weight_sum, 1e-8)
    merged_cam = np.clip(merged_cam, 0.0, None)
    max_value = float(merged_cam.max())
    if max_value > 1e-8:
        merged_cam = merged_cam / max_value

    return merged_cam


def _refine_hotspots(masked_cam: np.ndarray, confidence: float) -> np.ndarray:
    percentile = 82.0 if confidence >= 0.80 else 88.0
    hotspot_threshold = float(np.percentile(masked_cam, percentile))
    hotspot_mask = (masked_cam >= hotspot_threshold).astype(np.uint8)

    count, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(hotspot_mask)
    if count <= 1:
        return masked_cam

    peak_y, peak_x = np.unravel_index(int(np.argmax(masked_cam)), masked_cam.shape)
    peak_label = int(cc_labels[peak_y, peak_x])

    keep_ids: list[int] = []
    if peak_label > 0:
        keep_ids.append(peak_label)

    # For lower-confidence predictions, keep one additional large region.
    if confidence < 0.75:
        areas = cc_stats[1:, cv2.CC_STAT_AREA]
        sorted_ids = (np.argsort(areas)[::-1] + 1).tolist()
        for component_id in sorted_ids:
            if component_id not in keep_ids:
                keep_ids.append(int(component_id))
                break

    if not keep_ids:
        return masked_cam

    keep_mask = np.isin(cc_labels, keep_ids).astype(np.float32)
    keep_mask = cv2.GaussianBlur(keep_mask, (0, 0), sigmaX=1.0, sigmaY=1.0)
    refined = masked_cam * np.clip(keep_mask, 0.0, 1.0)

    refined = cv2.GaussianBlur(refined, (0, 0), sigmaX=0.8, sigmaY=0.8)
    max_value = float(refined.max())
    if max_value > 1e-8:
        refined = refined / max_value

    return refined


def generate_gradcam_image(
    image_input: Image.Image | bytes | bytearray,
    target_class_name: str | None = None,
    target_confidence: float | None = None,
) -> str | None:
    # Hybrid GradCAM: local-quality visualization with brain masking and no-tumor safeguards.
    try:
        model = _get_gradcam_model()
        input_tensor, rgb_image = _prepare_gradcam_inputs(image_input)

        with torch.inference_mode():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_index = int(probabilities.argmax(dim=1).item())

        target_index = predicted_index
        normalized_name = None
        if target_class_name:
            normalized_name = str(target_class_name).strip().lower()
            if normalized_name in CLASS_NAMES:
                target_index = CLASS_NAMES.index(normalized_name)

        brain_mask = _build_brain_mask(rgb_image)
        confidence = float(target_confidence or 0.0)

        # For no-tumor, avoid tumor-like hotspots and return a restrained cool map.
        if normalized_name == "notumor" and confidence >= 0.55:
            gray_uint8 = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            base_signal = cv2.normalize(gray_uint8.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
            base_signal = cv2.GaussianBlur(base_signal * brain_mask, (0, 0), sigmaX=1.2, sigmaY=1.2)

            blue = np.clip(0.28 + (0.62 * base_signal), 0.0, 1.0)
            green = np.clip(0.08 + (0.26 * base_signal), 0.0, 1.0)
            red = np.clip(0.03 + (0.10 * base_signal), 0.0, 1.0)
            cool_map = np.stack([red, green, blue], axis=-1)

            base_rgb = np.repeat(base_signal[..., None], 3, axis=2) * 0.35
            enhanced = np.clip((0.70 * cool_map) + (0.30 * base_rgb), 0.0, 1.0)
            enhanced = enhanced * (0.25 + 0.75 * brain_mask[..., None])
            enhanced_smooth = cv2.GaussianBlur((enhanced * 255.0).astype(np.uint8), (3, 3), sigmaX=0.5, sigmaY=0.5)
        else:
            grayscale_cam = _compute_ensemble_cam(model, input_tensor, target_index)
            masked_cam = np.clip(grayscale_cam * brain_mask, 0.0, None)
            masked_cam = np.power(masked_cam, 0.88)
            masked_cam = _refine_hotspots(masked_cam, confidence)

            peak = float(masked_cam.max())
            if peak > 1e-8:
                masked_cam = masked_cam / peak

            base_image = np.clip(
                rgb_image * (0.30 + (0.70 * brain_mask[..., None])),
                0.0,
                1.0,
            )
            cam_overlay = show_cam_on_image(
                base_image,
                np.clip(masked_cam, 0.0, 1.0),
                use_rgb=True,
                image_weight=0.36,
            )
            enhanced_smooth = cv2.GaussianBlur(cam_overlay, (3, 3), sigmaX=0.45, sigmaY=0.45)

        # Encode heatmap to base64 (in-memory, no disk I/O)
        heatmap_image = Image.fromarray(enhanced_smooth)
        buffer = io.BytesIO()
        heatmap_image.save(buffer, format="PNG")
        buffer.seek(0)
        import base64
        base64_image = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{base64_image}"
    except Exception as error:
        logger.exception("%s failed to generate GradCAM artifact", PLATFORM_NAME)
        return None


def coerce_feature_value(feature: str, value: Any, default: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid value type for cognitive feature '{feature}'",
        )

    if pd.isna(value):
        return default

    if isinstance(default, (int, float, np.integer, np.floating)):
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return default
            try:
                return float(stripped)
            except ValueError as error:
                raise HTTPException(
                    status_code=400,
                    detail=f"Feature '{feature}' must be numeric",
                ) from error

        raise HTTPException(
            status_code=400,
            detail=f"Feature '{feature}' must be numeric",
        )

    return str(value)


def prepare_cognitive_dataframe(
    artifacts: ANNArtifacts,
    payload: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if not isinstance(payload, dict) or not payload:
        raise HTTPException(
            status_code=400,
            detail="Clinical JSON payload must be a non-empty JSON object",
        )

    if _uses_compact_cognitive_schema(artifacts.cognitive_input_features):
        return prepare_compact_cognitive_dataframe(artifacts, payload)

    alias_map = build_cognitive_alias_map(artifacts.cognitive_input_features)
    aligned_row = dict(artifacts.cognitive_defaults)
    unknown_fields: list[str] = []
    recognized_fields = 0

    for key, value in payload.items():
        feature = alias_map.get(normalize_key(key))
        if feature is None:
            unknown_fields.append(key)
            continue

        aligned_row[feature] = coerce_feature_value(
            feature,
            value,
            artifacts.cognitive_defaults.get(feature),
        )
        recognized_fields += 1

    if recognized_fields == 0:
        raise HTTPException(
            status_code=400,
            detail="No recognized cognitive features were found in the input JSON",
        )

    ordered_row = {
        feature: aligned_row.get(feature, artifacts.cognitive_defaults.get(feature))
        for feature in artifacts.cognitive_input_features
    }
    dataframe = pd.DataFrame([ordered_row], columns=artifacts.cognitive_input_features)

    if unknown_fields:
        logger.warning(
            "%s ignored unknown cognitive fields: %s",
            PLATFORM_NAME,
            ", ".join(unknown_fields),
        )

    return dataframe, unknown_fields, {"mapped_inputs": ordered_row, "mapping_notes": {}}


def _extract_numeric_value(
    payload: dict[str, Any],
    *aliases: str,
) -> tuple[float | None, str | None]:
    normalized_payload = {
        normalize_key(str(key)): value
        for key, value in payload.items()
    }

    for alias in aliases:
        normalized_alias = normalize_key(alias)
        if normalized_alias not in normalized_payload:
            continue

        value = normalized_payload[normalized_alias]

        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue

        try:
            return float(value), normalized_alias
        except (TypeError, ValueError) as error:
            raise HTTPException(
                status_code=400,
                detail=f"Feature '{alias}' must be numeric",
            ) from error

    return None, None


def _extract_numeric_feature(
    payload: dict[str, Any],
    *aliases: str,
    default: float,
) -> float:
    value, _ = _extract_numeric_value(payload, *aliases)
    if value is None:
        return float(default)
    return float(value)


def _uses_compact_cognitive_schema(input_features: list[str]) -> bool:
    return len(input_features) == len(COGNITIVE_UI_MODEL_FEATURES) and set(input_features) == set(
        COGNITIVE_UI_MODEL_FEATURES
    )


def _map_education_years_to_level(education_years: float) -> float:
    clipped_years = float(_clamp(education_years, 0.0, COGNITIVE_UI_EDUCATION_MAX_YEARS))
    if clipped_years <= 8.0:
        return 0.0
    if clipped_years <= 12.0:
        return 1.0
    if clipped_years <= 16.0:
        return 2.0
    return COGNITIVE_EDUCATION_LEVEL_MAX


def _scale_ui_memory_score_to_mmse(memory_score: float) -> float:
    normalized_memory = _clamp(memory_score, 0.0, COGNITIVE_UI_MEMORY_MAX) / COGNITIVE_UI_MEMORY_MAX
    return float(normalized_memory * COGNITIVE_MMSE_MAX)


def _scale_ui_cognitive_score_to_functional(cognitive_score: float) -> float:
    normalized_cognitive = (
        _clamp(cognitive_score, 0.0, COGNITIVE_UI_COGNITIVE_MAX) / COGNITIVE_UI_COGNITIVE_MAX
    )
    return float(normalized_cognitive * COGNITIVE_FUNCTIONAL_MAX)


def prepare_compact_cognitive_dataframe(
    artifacts: ANNArtifacts,
    payload: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    allowed_keys = {
        "age",
        "education",
        "educationlevel",
        "mmse",
        "memoryscore",
        "memorytestscore",
        "functionalassessment",
        "cognitivescore",
    }
    unknown_fields = [
        str(key)
        for key in payload.keys()
        if normalize_key(str(key)) not in allowed_keys
    ]

    recognized_fields = 0
    mapped_inputs: dict[str, float] = {}
    mapping_notes: dict[str, str] = {}

    age_value, age_source = _extract_numeric_value(payload, "Age", "age")
    age_default = float(artifacts.cognitive_defaults.get("Age", 75.0))
    if age_value is None:
        age = age_default
        mapping_notes["Age"] = f"Age defaulted to {age:.1f} years from the training dataset median."
    else:
        recognized_fields += 1
        age = float(_clamp(age_value, COGNITIVE_AGE_MIN, COGNITIVE_AGE_MAX))
        if age != age_value:
            mapping_notes["Age"] = (
                f"Age {age_value:.1f} was clipped to {age:.1f} to stay inside the trained age range "
                f"({COGNITIVE_AGE_MIN:.0f}-{COGNITIVE_AGE_MAX:.0f})."
            )
        else:
            mapping_notes["Age"] = f"Age used directly at {age:.1f} years."
    mapped_inputs["Age"] = round(age, 4)

    education_level_value, education_level_source = _extract_numeric_value(
        payload,
        "EducationLevel",
        "education_level",
    )
    education_years_value, education_years_source = _extract_numeric_value(payload, "education")
    education_default = float(artifacts.cognitive_defaults.get("EducationLevel", 1.0))
    if education_level_value is not None:
        recognized_fields += 1
        education_level = float(_clamp(education_level_value, 0.0, COGNITIVE_EDUCATION_LEVEL_MAX))
        mapping_notes["EducationLevel"] = (
            f"Education level used directly at {education_level:.0f} on the training dataset scale."
        )
    elif education_years_value is not None:
        recognized_fields += 1
        education_level = _map_education_years_to_level(education_years_value)
        mapping_notes["EducationLevel"] = (
            f"Education years {education_years_value:.1f} were mapped to training level {education_level:.0f}."
        )
    else:
        education_level = education_default
        mapping_notes["EducationLevel"] = (
            f"Education defaulted to training level {education_level:.0f} from the dataset median."
        )
    mapped_inputs["EducationLevel"] = round(education_level, 4)

    mmse_value, mmse_source = _extract_numeric_value(payload, "MMSE", "mmse")
    memory_value, memory_source = _extract_numeric_value(payload, "memory_score", "memoryScore", "memorytestscore")
    mmse_default = float(artifacts.cognitive_defaults.get("MMSE", 14.0))
    if mmse_value is not None:
        recognized_fields += 1
        mmse = float(_clamp(mmse_value, 0.0, COGNITIVE_MMSE_MAX))
        mapping_notes["MMSE"] = f"MMSE used directly at {mmse:.1f}/{COGNITIVE_MMSE_MAX:.0f}."
    elif memory_value is not None:
        recognized_fields += 1
        mmse = _scale_ui_memory_score_to_mmse(memory_value)
        mapping_notes["MMSE"] = (
            f"Memory score {memory_value:.1f}/{COGNITIVE_UI_MEMORY_MAX:.0f} was aligned to MMSE "
            f"{mmse:.1f}/{COGNITIVE_MMSE_MAX:.0f}."
        )
    else:
        mmse = mmse_default
        mapping_notes["MMSE"] = f"MMSE defaulted to {mmse:.1f}/{COGNITIVE_MMSE_MAX:.0f} from the dataset median."
    mapped_inputs["MMSE"] = round(mmse, 4)

    functional_value, functional_source = _extract_numeric_value(
        payload,
        "FunctionalAssessment",
        "functionalassessment",
    )
    cognitive_value, cognitive_source = _extract_numeric_value(
        payload,
        "cognitive_score",
        "cognitiveScore",
    )
    functional_default = float(artifacts.cognitive_defaults.get("FunctionalAssessment", 5.0))
    if functional_value is not None:
        recognized_fields += 1
        functional_assessment = float(_clamp(functional_value, 0.0, COGNITIVE_FUNCTIONAL_MAX))
        mapping_notes["FunctionalAssessment"] = (
            f"Functional assessment used directly at {functional_assessment:.1f}/{COGNITIVE_FUNCTIONAL_MAX:.0f}."
        )
    elif cognitive_value is not None:
        recognized_fields += 1
        functional_assessment = _scale_ui_cognitive_score_to_functional(cognitive_value)
        mapping_notes["FunctionalAssessment"] = (
            f"Cognitive score {cognitive_value:.1f}/{COGNITIVE_UI_COGNITIVE_MAX:.0f} was aligned to "
            f"FunctionalAssessment {functional_assessment:.1f}/{COGNITIVE_FUNCTIONAL_MAX:.0f}."
        )
    else:
        functional_assessment = functional_default
        mapping_notes["FunctionalAssessment"] = (
            f"Functional assessment defaulted to {functional_assessment:.1f}/{COGNITIVE_FUNCTIONAL_MAX:.0f} "
            "from the dataset median."
        )
    mapped_inputs["FunctionalAssessment"] = round(functional_assessment, 4)

    if recognized_fields == 0:
        raise HTTPException(
            status_code=400,
            detail="No recognized cognitive features were found in the input JSON",
        )

    ordered_row = {
        feature: mapped_inputs.get(feature, float(artifacts.cognitive_defaults.get(feature, 0.0)))
        for feature in artifacts.cognitive_input_features
    }
    dataframe = pd.DataFrame([ordered_row], columns=artifacts.cognitive_input_features)
    context = {
        "mapped_inputs": mapped_inputs,
        "mapping_notes": mapping_notes,
        "input_sources": {
            "Age": age_source,
            "EducationLevel": education_level_source or education_years_source,
            "MMSE": mmse_source or memory_source,
            "FunctionalAssessment": functional_source or cognitive_source,
        },
    }
    return dataframe, unknown_fields, context

def _get_latest_cognitive_history() -> dict[str, Any] | None:
    with COGNITIVE_TEST_HISTORY_LOCK:
        if not COGNITIVE_TEST_HISTORY:
            return None
        latest = COGNITIVE_TEST_HISTORY[-1]

    module_scores = latest.get("module_scores") or {}
    serialized_scores: dict[str, float] = {}
    for key, value in module_scores.items():
        try:
            serialized_scores[str(key)] = float(value)
        except (TypeError, ValueError):
            continue

    return {
        "reliability_flag": str(latest.get("reliability_flag") or "standard").lower(),
        "module_scores": serialized_scores,
    }


def _compute_module_deficit() -> tuple[float, dict[str, float], str]:
    latest = _get_latest_cognitive_history()
    if latest is None:
        return 0.0, {}, "unavailable"

    module_scores = latest.get("module_scores") or {}
    module_deficit_breakdown: dict[str, float] = {}

    for module_name, max_score in COGNITIVE_MODULE_MAX_SCORES.items():
        score = float(module_scores.get(module_name, 0.0))
        normalized_score = _clamp(score / float(max_score), 0.0, 1.0)
        module_deficit_breakdown[module_name] = round(1.0 - normalized_score, 4)

    if not module_deficit_breakdown:
        return 0.0, {}, "unavailable"

    mean_deficit = float(sum(module_deficit_breakdown.values()) / len(module_deficit_breakdown))
    reliability_flag = str(latest.get("reliability_flag") or "standard").lower()
    reliability_weight = {
        "low": 0.75,
        "moderate": 0.90,
        "standard": 1.0,
    }.get(reliability_flag, 1.0)
    weighted_deficit = _clamp(mean_deficit * reliability_weight, 0.0, 1.0)

    return weighted_deficit, module_deficit_breakdown, reliability_flag


def _resolve_cognitive_risk_label(score: float) -> str:
    if score >= 0.66:
        return "High Risk"
    if score >= 0.33:
        return "Moderate Risk"
    return "Low Risk"


def _get_cognitive_feature_importances(artifacts: ANNArtifacts) -> dict[str, float]:
    if not hasattr(artifacts.cognitive_model, "named_steps"):
        return {}

    classifier = artifacts.cognitive_model.named_steps.get("classifier")
    preprocessor = artifacts.cognitive_model.named_steps.get("preprocessor")
    importances = getattr(classifier, "feature_importances_", None)
    if importances is None:
        return {}

    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        feature_names = [str(name) for name in preprocessor.get_feature_names_out().tolist()]
    else:
        feature_names = list(artifacts.cognitive_input_features)

    feature_importances: dict[str, float] = {}
    for feature_name, importance in zip(feature_names, importances):
        raw_feature_name = str(feature_name).split("__", 1)[-1]
        feature_importances[raw_feature_name] = feature_importances.get(raw_feature_name, 0.0) + float(
            importance
        )

    total_importance = float(sum(feature_importances.values()))
    if total_importance <= 0.0:
        return {}

    return {
        feature_name: round(float(importance / total_importance), 6)
        for feature_name, importance in feature_importances.items()
    }


def _build_cognitive_contributing_factors(
    artifacts: ANNArtifacts,
    cognitive_context: dict[str, Any],
    model_probability: float,
) -> list[dict[str, Any]]:
    mapped_inputs = cognitive_context.get("mapped_inputs") or {}
    mapping_notes = cognitive_context.get("mapping_notes") or {}
    feature_importances = _get_cognitive_feature_importances(artifacts)
    if not feature_importances and mapped_inputs:
        equal_weight = 1.0 / float(len(mapped_inputs))
        feature_importances = {
            str(feature_name): equal_weight
            for feature_name in mapped_inputs.keys()
        }

    feature_specs = {
        "Age": {
            "label": "Age-aligned risk signal",
            "minimum": COGNITIVE_AGE_MIN,
            "maximum": COGNITIVE_AGE_MAX,
            "reverse": False,
        },
        "EducationLevel": {
            "label": "Education-level risk signal",
            "minimum": 0.0,
            "maximum": COGNITIVE_EDUCATION_LEVEL_MAX,
            "reverse": True,
        },
        "MMSE": {
            "label": "Memory score risk signal",
            "minimum": 0.0,
            "maximum": COGNITIVE_MMSE_MAX,
            "reverse": True,
        },
        "FunctionalAssessment": {
            "label": "Cognitive function risk signal",
            "minimum": 0.0,
            "maximum": COGNITIVE_FUNCTIONAL_MAX,
            "reverse": True,
        },
    }

    factors: list[dict[str, Any]] = []
    for feature_name, spec in feature_specs.items():
        if feature_name not in mapped_inputs:
            continue

        mapped_value = float(mapped_inputs.get(feature_name) or 0.0)
        feature_importance = float(feature_importances.get(feature_name, 0.0))
        value_range = max(float(spec["maximum"]) - float(spec["minimum"]), 1e-6)
        if spec["reverse"]:
            risk_component = _clamp((float(spec["maximum"]) - mapped_value) / value_range, 0.0, 1.0)
        else:
            risk_component = _clamp((mapped_value - float(spec["minimum"])) / value_range, 0.0, 1.0)

        factors.append(
            {
                "factor": spec["label"],
                "feature": feature_name,
                "weight": round(feature_importance, 4),
                "value": round(risk_component, 4),
                "mapped_value": round(mapped_value, 4),
                "impact_score": round(feature_importance * risk_component, 4),
                "detail": str(mapping_notes.get(feature_name) or f"{feature_name} aligned to the training schema."),
            }
        )

    if not factors:
        return [
            {
                "factor": "Trained cognitive model probability",
                "weight": 1.0,
                "value": round(model_probability, 4),
                "impact_score": round(model_probability, 4),
                "detail": f"Model probability based on the trained cognitive classifier: {model_probability * 100.0:.1f}%.",
            }
        ]

    factors.sort(key=lambda item: item["impact_score"], reverse=True)
    return factors


def predict_cognitive_risk(
    artifacts: ANNArtifacts,
    payload: dict[str, Any],
) -> dict[str, Any]:
    input_frame, unknown_fields, cognitive_context = prepare_cognitive_dataframe(artifacts, payload)

    with artifacts.cognitive_lock:
        prediction = artifacts.cognitive_model.predict(input_frame)[0]
        probability = None

        if hasattr(artifacts.cognitive_model, "predict_proba"):
            probabilities = artifacts.cognitive_model.predict_proba(input_frame)[0]
            classes = list(getattr(artifacts.cognitive_model, "classes_", []))
            if 1 in classes:
                probability = float(probabilities[classes.index(1)])
            elif len(probabilities) > 0:
                probability = float(probabilities[-1])

    model_probability = probability if probability is not None else float(int(prediction))
    model_probability = _clamp(float(model_probability), 0.0, 1.0)

    _, module_deficit_breakdown, module_reliability_flag = _compute_module_deficit()
    cognitive_risk = _resolve_cognitive_risk_label(model_probability)
    top_factors = _build_cognitive_contributing_factors(
        artifacts,
        cognitive_context,
        model_probability,
    )

    logger.info(
        "%s cognitive prediction complete: risk=%s probability=%.4f",
        PLATFORM_NAME,
        cognitive_risk,
        model_probability,
    )
    return {
        "cognitive_risk": cognitive_risk,
        "risk_label": cognitive_risk,
        "risk_score": round(model_probability, 4),
        "risk_probability": round(model_probability, 4),
        "risk_percentage": round(model_probability * 100.0, 2),
        "model_probability": round(model_probability, 4),
        "hybrid_components": {"model_probability": round(model_probability, 4)},
        "aligned_inputs": cognitive_context.get("mapped_inputs") or {},
        "score_source": "trained_cognitive_model",
        "module_deficit_breakdown": module_deficit_breakdown,
        "module_reliability_flag": module_reliability_flag,
        "top_contributing_factors": top_factors,
        "unknown_fields": unknown_fields,
    }


def _resolve_final_risk_category(score: float) -> str:
    if score >= 0.80:
        return "Very High"
    if score >= 0.60:
        return "High"
    if score >= 0.35:
        return "Moderate"
    return "Low"


def _build_multimodal_contributing_factors(
    mri_result: dict[str, Any],
    cognitive_result: dict[str, Any],
) -> list[dict[str, Any]]:
    factors: list[dict[str, Any]] = []

    mri_tumor_probability = _clamp(float(mri_result.get("tumor_probability") or 0.0), 0.0, 1.0)
    cognitive_probability = _clamp(float(cognitive_result.get("risk_score") or 0.0), 0.0, 1.0)

    factors.append(
        {
            "factor": "MRI tumor burden probability",
            "impact_score": round(FUSION_MRI_WEIGHT * mri_tumor_probability, 4),
            "detail": f"Tumor-associated MRI probability: {mri_tumor_probability * 100.0:.1f}%",
        }
    )
    factors.append(
        {
            "factor": "Cognitive model risk",
            "impact_score": round(FUSION_COGNITIVE_WEIGHT * cognitive_probability, 4),
            "detail": f"Model-based cognitive risk score: {cognitive_probability * 100.0:.1f}%",
        }
    )
    cognitive_factors = cognitive_result.get("top_contributing_factors") or []
    for factor in cognitive_factors[:3]:
        factors.append(
            {
                "factor": str(factor.get("factor") or "Cognitive factor"),
                "impact_score": round(float(factor.get("impact_score") or 0.0), 4),
                "detail": str(
                    factor.get("detail")
                    or (
                        f"Component value {float(factor.get('value') or 0.0):.2f} "
                        f"with weight {float(factor.get('weight') or 0.0):.2f}"
                    )
                ),
            }
        )

    factors.sort(key=lambda item: item["impact_score"], reverse=True)
    return factors[:5]


def build_ann_summary(
    mri_result: dict[str, Any],
    cognitive_result: dict[str, Any],
    final_risk_score: float,
    final_risk_category: str,
) -> str:
    tumor_prediction = str(mri_result.get("tumor_prediction") or "notumor").strip().lower()
    confidence_level = str(mri_result.get("confidence_level") or "Moderate").lower()
    calibrated_confidence = float(mri_result.get("tumor_confidence") or 0.0)

    if tumor_prediction == "notumor":
        mri_statement = (
            "The MRI model indicates no dominant tumor pattern "
            f"with {confidence_level} confidence ({calibrated_confidence * 100.0:.1f}%)."
        )
    else:
        mri_statement = (
            f"The MRI model favors a {tumor_prediction} pattern with {confidence_level} confidence "
            f"({calibrated_confidence * 100.0:.1f}%)."
        )

    cognitive_risk = str(cognitive_result.get("cognitive_risk") or "Low Risk")
    cognitive_probability = float(cognitive_result.get("risk_score") or 0.0)
    cognitive_statement = (
        f"The trained cognitive model indicates {cognitive_risk.lower()} "
        f"({cognitive_probability * 100.0:.1f}%)."
    )

    fusion_statement = (
        f"Multimodal fusion (MRI {int(FUSION_MRI_WEIGHT * 100)}% + cognitive model {int(FUSION_COGNITIVE_WEIGHT * 100)}%) "
        f"estimates an overall brain-health risk of {final_risk_score * 100.0:.1f}% "
        f"({final_risk_category.lower()})."
    )
    return " ".join(
        [
            mri_statement,
            "GradCAM localizes regions that influenced this MRI decision.",
            cognitive_statement,
            fusion_statement,
        ]
    )


def run_full_analysis(
    artifacts: ANNArtifacts,
    image_bytes: bytes,
    filename: str,
    content_type: str | None,
    cognitive_payload: dict[str, Any],
) -> dict[str, Any]:
    image_for_prediction = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    mri_result, prepared_image = predict_mri_image_from_image(image_for_prediction, filename)

    gradcam_image = generate_gradcam_image(
        prepared_image,
        mri_result.get("prediction"),
        mri_result.get("confidence"),
    )

    cognitive_result = predict_cognitive_risk(artifacts, cognitive_payload)
    cognitive_score = float(cognitive_result.get("risk_score") or 0.0)
    tumor_probability = float(mri_result.get("tumor_probability") or 0.0)
    final_risk_score = _clamp(
        (FUSION_MRI_WEIGHT * tumor_probability)
        + (FUSION_COGNITIVE_WEIGHT * cognitive_score),
        0.0,
        1.0,
    )
    final_risk_category = _resolve_final_risk_category(final_risk_score)
    top_contributing_factors = _build_multimodal_contributing_factors(mri_result, cognitive_result)

    report_summary = build_ann_summary(
        mri_result,
        cognitive_result,
        final_risk_score,
        final_risk_category,
    )

    rounded_confidence = round(float(mri_result.get("tumor_confidence") or 0.0), 4)
    rounded_risk_score = round(cognitive_score, 4)
    rounded_final_risk = round(final_risk_score, 4)

    return {
        "platform": PLATFORM_NAME,
        "tumor_prediction": str(mri_result.get("tumor_prediction") or "notumor"),
        "confidence": rounded_confidence,
        "tumor_confidence": rounded_confidence,
        "raw_confidence": round(float(mri_result.get("raw_confidence") or 0.0), 4),
        "mri_uncertainty": round(float(mri_result.get("mri_uncertainty") or 0.0), 4),
        "uncertainty_score": round(float(mri_result.get("uncertainty_score") or 0.0), 4),
        "uncertainty_variance": round(float(mri_result.get("uncertainty_variance") or 0.0), 6),
        "uncertainty_entropy": round(float(mri_result.get("uncertainty_entropy") or 0.0), 4),
        "confidence_level": str(mri_result.get("confidence_level") or "Moderate"),
        "temperature": round(float(mri_result.get("temperature") or MRI_TEMPERATURE), 3),
        "tumor_probability": round(float(mri_result.get("tumor_probability") or 0.0), 4),
        "notumor_probability": round(float(mri_result.get("notumor_probability") or 0.0), 4),
        "class_probabilities": mri_result.get("class_probabilities") or {},
        "cognitive_risk": str(cognitive_result.get("cognitive_risk") or "Low Risk"),
        "risk_label": str(cognitive_result.get("risk_label") or "Low Risk"),
        "risk_score": rounded_risk_score,
        "risk_probability": rounded_risk_score,
        "risk_percentage": round(float(cognitive_result.get("risk_percentage") or 0.0), 2),
        "cognitive_model_probability": round(float(cognitive_result.get("model_probability") or 0.0), 4),
        "hybrid_components": cognitive_result.get("hybrid_components") or {},
        "module_deficit_breakdown": cognitive_result.get("module_deficit_breakdown") or {},
        "module_reliability_flag": str(cognitive_result.get("module_reliability_flag") or "unavailable"),
        "final_risk_score": rounded_final_risk,
        "final_risk_percent": round(final_risk_score * 100.0, 2),
        "final_risk_category": final_risk_category,
        "top_contributing_factors": top_contributing_factors,
        "explainability": {
            "top_factors": top_contributing_factors,
            "cognitive_factors": cognitive_result.get("top_contributing_factors") or [],
            "mri_confidence_level": str(mri_result.get("confidence_level") or "Moderate"),
            "mri_uncertainty": round(float(mri_result.get("mri_uncertainty") or 0.0), 4),
        },
        "gradcam_image": gradcam_image,
        "report_summary": report_summary,
        "summary": report_summary,
        "clinical_disclaimer": CLINICAL_DISCLAIMER,
    }


def get_artifacts(request: Request) -> ANNArtifacts:
    artifacts = getattr(request.app.state, "ann", None)
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Neuro Vision models are not loaded")
    return artifacts


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    content = exc.detail if isinstance(exc.detail, dict) else {"detail": exc.detail}
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("%s internal error while handling %s", PLATFORM_NAME, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Neuro Vision backend encountered an internal error"},
    )


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Neuro Vision Multimodal AI Backend Running"}


@app.post("/predict_mri")
@app.post("/api/prediction/mri")
async def predict_mri(request: Request, file: UploadFile = File(...)) -> Any:
    _ = get_artifacts(request)
    
    try:
        validated_image, _ = await read_uploaded_image(file)
    except HTTPException as e:
        logger.warning("MRI prediction rejected: %s", e.detail)
        return JSONResponse(
            status_code=400,
            content={"status": "rejected", "message": str(e.detail)}
        )

    mri_result, _ = predict_mri_image_from_image(validated_image, file.filename or "image.png")

    return mri_result


@app.post("/predict_cognitive")
@app.post("/api/prediction/cognitive")
async def predict_cognitive(
    request: Request,
    payload: dict[str, Any],
) -> dict[str, Any]:
    artifacts = get_artifacts(request)
    result = predict_cognitive_risk(artifacts, payload)

    return dict(result)


@app.post("/gradcam")
async def gradcam(request: Request, file: UploadFile = File(...)) -> Any:
    _ = get_artifacts(request)
    
    try:
        validated_image, image_bytes = await read_uploaded_image(file)
    except HTTPException as e:
        logger.warning("GradCAM rejected: %s", e.detail)
        return JSONResponse(
            status_code=400,
            content={"status": "rejected", "message": str(e.detail)}
        )

    prepared_image = validated_image.convert("RGB")
    mri_prediction = _resolve_gradcam_target_prediction(
        prepared_image,
        file.filename or "image.png",
    )

    heatmap = generate_gradcam_image(
        prepared_image,
        mri_prediction.get("prediction"),
        mri_prediction.get("confidence"),
    )

    if not heatmap:
        raise HTTPException(status_code=500, detail="Unable to generate GradCAM heatmap")

    return {
        "prediction": mri_prediction.get("prediction"),
        "confidence": float(mri_prediction.get("confidence") or 0.0),
        "prediction_source": mri_prediction.get("prediction_source", "unknown"),
        "heatmap": heatmap,
        "gradcam_image": heatmap,
    }


@app.post("/analyze")
@app.post("/api/prediction/final-risk")
async def analyze(
    request: Request,
    file: UploadFile | None = File(None),
    clinical_json: str | None = Form(None),
) -> Any:
    artifacts = get_artifacts(request)

    if file is None:
        return JSONResponse(
            status_code=400,
            content={"status": "rejected", "message": "No image provided."}
        )

    try:
        _, image_bytes = await read_uploaded_image(file)
        cognitive_payload = await resolve_clinical_payload(request, clinical_json)
        return run_full_analysis(
            artifacts,
            image_bytes,
            file.filename or "image.png",
            file.content_type,
            cognitive_payload,
        )
    except HTTPException as e:
        logger.warning("Analysis rejected: %s", e.detail)
        return JSONResponse(
            status_code=e.status_code,
            content={"status": "rejected", "message": str(e.detail)}
        )
    except Exception as error:
        logger.exception("%s failed to generate final risk report", PLATFORM_NAME)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Unable to generate report: {error}"}
        )


@app.post("/ai_report")
@app.post("/api/prediction/report")
async def ai_report(
    request: Request,
    file: UploadFile = File(...),
    clinical_json: str = Form(...),
) -> Any:
    artifacts = get_artifacts(request)
    
    try:
        _, image_bytes = await read_uploaded_image(file)
        cognitive_payload = await resolve_clinical_payload(request, clinical_json)
        return run_full_analysis(
            artifacts,
            image_bytes,
            file.filename or "image.png",
            file.content_type,
            cognitive_payload,
        )
    except HTTPException as e:
        logger.warning("AI Report rejected: %s", e.detail)
        return JSONResponse(
            status_code=e.status_code,
            content={"status": "rejected", "message": str(e.detail)}
        )
    except Exception as error:
        logger.exception("%s failed to generate AI report", PLATFORM_NAME)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Unable to generate report: {error}"}
        )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _resolve_existing_directory(primary: Path, fallback: Path) -> Path | None:
    if primary.is_dir():
        return primary
    if fallback.is_dir():
        return fallback
    return None


def _count_mri_samples_by_class(directory: Path | None) -> dict[str, int]:
    if directory is None:
        return {class_name: 0 for class_name in CLASS_NAMES}

    discovered_counts: dict[str, int] = {}
    for class_dir in directory.iterdir():
        if not class_dir.is_dir():
            continue
        image_count = sum(
            1
            for file_path in class_dir.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
        )
        discovered_counts[class_dir.name.lower()] = int(image_count)

    return {
        class_name: int(discovered_counts.get(class_name, 0))
        for class_name in CLASS_NAMES
    }


def _resolve_cognitive_target_column(dataframe: pd.DataFrame) -> str | None:
    column_lookup = {column.lower(): column for column in dataframe.columns}
    for alias in COGNITIVE_TARGET_ALIASES:
        if alias in dataframe.columns:
            return alias
        alias_lower = alias.lower()
        if alias_lower in column_lookup:
            return column_lookup[alias_lower]
    return None


@app.get("/api/datasets/overview")
async def datasets_overview() -> dict[str, Any]:
    train_dir = _resolve_existing_directory(PRIMARY_MRI_TRAIN_DIR, FALLBACK_MRI_TRAIN_DIR)
    test_dir = _resolve_existing_directory(PRIMARY_MRI_TEST_DIR, FALLBACK_MRI_TEST_DIR)

    train_counts = _count_mri_samples_by_class(train_dir)
    test_counts = _count_mri_samples_by_class(test_dir)

    cognitive_path = resolve_cognitive_data_path()
    cognitive_rows = 0
    cognitive_columns: list[str] = []

    if cognitive_path is not None:
        try:
            dataframe = pd.read_csv(cognitive_path)
            cognitive_rows = int(len(dataframe))
            cognitive_columns = [str(column) for column in dataframe.columns.tolist()]
        except Exception:
            logger.warning("%s failed to read cognitive dataset for overview", PLATFORM_NAME)

    return {
        "mri": {
            "training": {
                "path": str(train_dir) if train_dir is not None else None,
                "class_counts": train_counts,
                "total": int(sum(train_counts.values())),
            },
            "testing": {
                "path": str(test_dir) if test_dir is not None else None,
                "class_counts": test_counts,
                "total": int(sum(test_counts.values())),
            },
        },
        "cognitive": {
            "path": str(cognitive_path) if cognitive_path is not None else None,
            "rows": cognitive_rows,
            "columns": cognitive_columns,
        },
    }


@app.get("/api/datasets/cognitive-profile")
async def datasets_cognitive_profile() -> dict[str, Any]:
    cognitive_path = resolve_cognitive_data_path()
    if cognitive_path is None:
        raise HTTPException(status_code=404, detail="Cognitive dataset not found")

    try:
        dataframe = pd.read_csv(cognitive_path)
    except Exception as error:
        raise HTTPException(status_code=500, detail="Failed to load cognitive dataset") from error

    target_column = _resolve_cognitive_target_column(dataframe)
    target_distribution: dict[str, int] = {}
    if target_column is not None:
        distribution_series = dataframe[target_column].value_counts(dropna=False)
        target_distribution = {
            str(index): int(value)
            for index, value in distribution_series.items()
        }

    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    age_stats: dict[str, float] = {}
    if "Age" in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe["Age"]):
        age_series = dataframe["Age"].dropna()
        if not age_series.empty:
            age_stats = {
                "min": round(float(age_series.min()), 2),
                "median": round(float(age_series.median()), 2),
                "max": round(float(age_series.max()), 2),
            }

    return {
        "path": str(cognitive_path),
        "rows": int(len(dataframe)),
        "columns": [str(column) for column in dataframe.columns.tolist()],
        "numeric_columns": [str(column) for column in numeric_columns],
        "target_column": target_column,
        "target_distribution": target_distribution,
        "age_stats": age_stats,
    }


@app.get("/api/cognitive/word-bank")
async def cognitive_word_bank() -> dict[str, Any]:
    return {
        "words": COGNITIVE_WORD_BANK,
        "encoding_size": 6,
    }


@app.get("/api/cognitive/reasoning-questions")
async def cognitive_reasoning_questions() -> dict[str, Any]:
    return {
        "questions": COGNITIVE_REASONING_QUESTIONS,
        "stroop_trials": COGNITIVE_STROOP_TRIALS,
        "symbol_digit_trials": COGNITIVE_SYMBOL_DIGIT_TRIALS,
        "mental_arithmetic": COGNITIVE_MENTAL_ARITHMETIC_BANK,
    }


@app.get("/api/cognitive/category-questions")
async def cognitive_category_questions() -> dict[str, Any]:
    return {
        "questions": COGNITIVE_CATEGORY_QUESTIONS,
        "animal_lexicon": COGNITIVE_ANIMAL_LEXICON,
    }


@app.get("/api/cognitive/spatial-rotation")
async def cognitive_spatial_rotation_questions() -> dict[str, Any]:
    return {
        "questions": COGNITIVE_SPATIAL_ROTATION_QUESTIONS,
        "visual_patterns": COGNITIVE_VISUAL_PATTERN_BANK,
    }


class CognitiveTestResultsRequest(BaseModel):
    total_duration_seconds: float = 0.0
    orientation_score: int = 0
    digit_span_forward_correct: bool = False
    digit_span_backward_correct: bool = False
    executive_reasoning_correct: bool = False
    reaction_average_ms: float = 9999.0
    reaction_accuracy_rate: float = 0.0
    reaction_missed_targets: int = 0
    visual_pattern_correct: int = 0
    verbal_fluency_unique_animals: int = 0
    category_matching_correct: bool = False
    stroop_correct: bool = False
    symbol_digit_correct: bool = False
    mental_arithmetic_correct_count: int = 0
    spatial_rotation_correct: bool = False
    delayed_recall_words: list[str] | None = None
    encoded_words: list[str] | None = None


@app.post("/cognitive-test-results")
async def cognitive_test_results(payload: CognitiveTestResultsRequest) -> dict[str, Any]:
    orientation_score = int(_clamp(float(payload.orientation_score), 0.0, 3.0))
    digit_forward_score = 2 if payload.digit_span_forward_correct else 0
    digit_backward_score = 2 if payload.digit_span_backward_correct else 0
    executive_reasoning_score = 2 if payload.executive_reasoning_correct else 0

    reaction_average_ms = max(0.0, float(payload.reaction_average_ms or 0.0))
    reaction_accuracy_rate = _clamp(float(payload.reaction_accuracy_rate or 0.0), 0.0, 1.0)
    if reaction_average_ms < 300.0:
        reaction_time_score = 2.0
    elif reaction_average_ms <= 500.0:
        reaction_time_score = 1.0
    else:
        reaction_time_score = 0.0
    reaction_score = round(reaction_time_score * reaction_accuracy_rate, 3)

    visual_pattern_score = int(_clamp(float(payload.visual_pattern_correct), 0.0, 3.0))
    verbal_fluency_count = max(0, int(payload.verbal_fluency_unique_animals))
    if verbal_fluency_count <= 3:
        verbal_fluency_score = 0
    elif verbal_fluency_count <= 7:
        verbal_fluency_score = 1
    else:
        verbal_fluency_score = 2

    category_score = 2 if payload.category_matching_correct else 0
    stroop_score = 2 if payload.stroop_correct else 0
    symbol_digit_score = 2 if payload.symbol_digit_correct else 0
    mental_arithmetic_score = int(_clamp(float(payload.mental_arithmetic_correct_count), 0.0, 2.0))
    spatial_rotation_score = 2 if payload.spatial_rotation_correct else 0

    encoded_source = [
        str(word).strip().lower()
        for word in (payload.encoded_words or COGNITIVE_WORD_BANK[:6])
        if str(word).strip()
    ]
    encoded_unique = list(dict.fromkeys(encoded_source))
    encoded_word_count = len(encoded_unique) if encoded_unique else 6
    encoded_set = set(encoded_unique)

    delayed_recall_words = {
        str(word).strip().lower()
        for word in (payload.delayed_recall_words or [])
        if str(word).strip()
    }
    delayed_recall_count = min(len(delayed_recall_words.intersection(encoded_set)), 6)

    memory_score = round((delayed_recall_count / float(encoded_word_count)) * 25.0, 2)
    memory_score = _clamp(memory_score, 0.0, 25.0)

    attention_domain = _clamp(
        (orientation_score / 3.0) * 0.5
        + (digit_forward_score / 2.0) * 0.3
        + (stroop_score / 2.0) * 0.2,
        0.0,
        1.0,
    )
    working_memory_domain = _clamp(
        (digit_backward_score / 2.0) * 0.4
        + (mental_arithmetic_score / 2.0) * 0.3
        + (delayed_recall_count / 6.0) * 0.3,
        0.0,
        1.0,
    )
    executive_domain = _clamp(
        (executive_reasoning_score / 2.0) * 0.6 + (category_score / 2.0) * 0.4,
        0.0,
        1.0,
    )
    processing_domain = _clamp(
        (reaction_score / 2.0) * 0.6 + (symbol_digit_score / 2.0) * 0.4,
        0.0,
        1.0,
    )
    language_domain = _clamp(
        (verbal_fluency_score / 2.0) * 0.7 + (delayed_recall_count / 6.0) * 0.3,
        0.0,
        1.0,
    )
    visuospatial_domain = _clamp(
        (visual_pattern_score / 3.0) * 0.5 + (spatial_rotation_score / 2.0) * 0.5,
        0.0,
        1.0,
    )

    cognitive_fraction = (
        (0.20 * attention_domain)
        + (0.20 * working_memory_domain)
        + (0.15 * executive_domain)
        + (0.15 * processing_domain)
        + (0.15 * language_domain)
        + (0.15 * visuospatial_domain)
    )
    cognitive_score = round(_clamp(cognitive_fraction * 30.0, 0.0, 30.0), 2)

    duration_seconds = max(0.0, float(payload.total_duration_seconds or 0.0))
    duration_minutes = round(duration_seconds / 60.0, 2)
    if duration_seconds < 300.0:
        reliability_flag = "low"
        reliability_message = (
            "Low reliability: test was completed in under 5 minutes. "
            "Retake is recommended for a clinically useful score."
        )
    elif duration_seconds < 480.0 or duration_seconds > 720.0:
        reliability_flag = "moderate"
        reliability_message = (
            "Completion time was outside the expected 8-12 minute window. "
            "Interpret with caution."
        )
    else:
        reliability_flag = "standard"
        reliability_message = "Completion time is within the expected range."

    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "memory_score": memory_score,
        "cognitive_score": cognitive_score,
        "duration_minutes": duration_minutes,
        "reliability_flag": reliability_flag,
        "module_scores": {
            "orientation": orientation_score,
            "digit_forward": digit_forward_score,
            "digit_backward": digit_backward_score,
            "executive_reasoning": executive_reasoning_score,
            "reaction": reaction_score,
            "visual_pattern": visual_pattern_score,
            "verbal_fluency": verbal_fluency_score,
            "category": category_score,
            "stroop": stroop_score,
            "symbol_digit": symbol_digit_score,
            "mental_arithmetic": mental_arithmetic_score,
            "spatial_rotation": spatial_rotation_score,
            "delayed_recall": delayed_recall_count,
        },
    }

    with COGNITIVE_TEST_HISTORY_LOCK:
        COGNITIVE_TEST_HISTORY.append(record)
        if len(COGNITIVE_TEST_HISTORY) > COGNITIVE_TEST_HISTORY_LIMIT:
            del COGNITIVE_TEST_HISTORY[:-COGNITIVE_TEST_HISTORY_LIMIT]

    logger.info("%s cognitive screening result stored", PLATFORM_NAME)
    return {
        "memory_score": memory_score,
        "cognitive_score": cognitive_score,
        "duration_minutes": duration_minutes,
        "reliability_flag": reliability_flag,
        "reliability_message": reliability_message,
        "module_scores": record["module_scores"],
        "delayed_recall_count": delayed_recall_count,
        "encoded_word_count": encoded_word_count,
    }


# ── PDF REPORT ────────────────────────────────────────────────────────────────

class ReportRequest(BaseModel):
    tumor_prediction: str = ""
    confidence: float = 0.0
    tumor_confidence: float | None = None
    cognitive_risk: str = ""
    risk_score: float | None = None
    risk_probability: float | None = None
    report_summary: str | None = None
    summary: str | None = None
    gradcam_image: str | None = None


def _build_pdf_report(
    tumor_prediction: str,
    confidence: float,
    cognitive_risk: str,
    risk_score: float,
    report_summary: str,
    gradcam_base64: str | None,
) -> bytes:
    buffer = io.BytesIO()

    text_color = HexColor("#000000")
    heading_color = HexColor("#333333")
    divider_color = HexColor("#1F5FA8")

    page_width, _page_height = A4
    margin = 2.54 * cm
    content_width = page_width - (2 * margin)

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
        title="Neuro Vision AI Brain Health Diagnostic Report",
        author="Neuro Vision AI Platform",
    )

    def ps(name: str, **kwargs: Any) -> ParagraphStyle:
        return ParagraphStyle(name, **kwargs)

    styles = {
        "header_brand": ps(
            "header_brand",
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=heading_color,
            alignment=1,
            spaceAfter=2,
        ),
        "header_title": ps(
            "header_title",
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=heading_color,
            alignment=1,
            spaceAfter=4,
        ),
        "header_subtitle": ps(
            "header_subtitle",
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=text_color,
            alignment=1,
            spaceAfter=4,
        ),
        "section_heading": ps(
            "section_heading",
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=heading_color,
            spaceBefore=2,
            spaceAfter=5,
        ),
        "label": ps(
            "label",
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=text_color,
            spaceAfter=0,
        ),
        "value": ps(
            "value",
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=text_color,
            spaceAfter=0,
        ),
        "body": ps(
            "body",
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=text_color,
            spaceAfter=3,
        ),
        "bullet": ps(
            "bullet",
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=text_color,
            leftIndent=8,
            spaceAfter=2,
        ),
        "caption": ps(
            "caption",
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=text_color,
            alignment=1,
            spaceAfter=4,
        ),
        "footer_heading": ps(
            "footer_heading",
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=heading_color,
            alignment=1,
            spaceAfter=3,
        ),
        "footer_text": ps(
            "footer_text",
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=text_color,
            alignment=1,
            spaceAfter=2,
        ),
    }

    generated_at = datetime.datetime.now()
    report_id = f"NV-{generated_at.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

    normalized_tumor = (tumor_prediction or "").strip().lower()
    if normalized_tumor == "notumor":
        predicted_tumor_type = "No Tumor Detected"
    elif normalized_tumor:
        predicted_tumor_type = tumor_prediction.replace("_", " ").strip().title()
    else:
        predicted_tumor_type = "Not Available"

    confidence_value = float(confidence)
    if confidence_value > 1.0:
        confidence_value = confidence_value / 100.0 if confidence_value <= 100.0 else 1.0
    confidence_value = max(0.0, min(confidence_value, 1.0))
    confidence_text = f"{confidence_value * 100:.1f}%"

    risk_value = float(risk_score)
    if risk_value > 1.0:
        risk_value = risk_value / 100.0 if risk_value <= 100.0 else 1.0
    risk_value = max(0.0, min(risk_value, 1.0))
    risk_score_text = f"{risk_value:.2f}"

    risk_label = (cognitive_risk or "").replace("_", " ").strip()
    if risk_label:
        risk_label = risk_label.title()
    else:
        risk_label = "Not Available"

    summary_text = (report_summary or "").strip()
    if not summary_text:
        tumor_clause = (
            "no convincing tumor-related abnormality"
            if predicted_tumor_type == "No Tumor Detected"
            else f"the presence of a {predicted_tumor_type.lower()} pattern"
        )
        summary_text = (
            f"The Neuro Vision AI analysis suggests {tumor_clause} with high model confidence ({confidence_text}). "
            "GradCAM visualization indicates the regions influencing the classification decision. "
            f"Cognitive risk indicators suggest a {risk_label.lower()} profile with a risk score of {risk_score_text}."
        )
    safe_summary = html.escape(summary_text).replace("\n", "<br/>")

    story: list[Any] = []

    def add_section_heading(title: str) -> None:
        story.append(Spacer(1, 0.18 * cm))
        story.append(HRFlowable(width="100%", thickness=0.6, color=divider_color, spaceBefore=2, spaceAfter=6))
        story.append(Paragraph(title, styles["section_heading"]))

    # Header
    story.extend(
        [
            Paragraph("NEURO VISION", styles["header_brand"]),
            Paragraph("AI Brain Health Diagnostic Report", styles["header_title"]),
            Paragraph("AI-assisted multimodal neurological analysis", styles["header_subtitle"]),
            HRFlowable(width="100%", thickness=0.8, color=divider_color, spaceBefore=2, spaceAfter=8),
        ]
    )

    # Report Information
    add_section_heading("Report Information")
    info_table = Table(
        [
            [Paragraph("Report ID:", styles["label"]), Paragraph(report_id, styles["value"])],
            [
                Paragraph("Date of Analysis:", styles["label"]),
                Paragraph(generated_at.strftime("%B %d, %Y %H:%M"), styles["value"]),
            ],
            [Paragraph("System Version:", styles["label"]), Paragraph("Neuro Vision v1.0", styles["value"])],
            [Paragraph("Generated By:", styles["label"]), Paragraph("Neuro Vision AI Platform", styles["value"])],
        ],
        colWidths=[4.1 * cm, content_width - (4.1 * cm)],
        hAlign="LEFT",
    )
    info_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(info_table)

    # MRI Tumor Classification
    add_section_heading("MRI Tumor Classification")
    story.append(Paragraph(f"- Predicted Tumor Type: {predicted_tumor_type}", styles["bullet"]))
    story.append(Paragraph(f"- Confidence Score: {confidence_text}", styles["bullet"]))

    # AI Explainability
    add_section_heading("AI Explainability (GradCAM)")
    if gradcam_base64:
        try:
            import base64
            # Decode base64 to bytes
            gradcam_bytes = base64.b64decode(gradcam_base64)
            # Create PIL Image from bytes
            gradcam_pil_image = Image.open(io.BytesIO(gradcam_bytes))
            # Save to BytesIO for RLImage
            gradcam_buffer = io.BytesIO()
            gradcam_pil_image.save(gradcam_buffer, format="PNG")
            gradcam_buffer.seek(0)
            # Embed in PDF
            gradcam_image = RLImage(gradcam_buffer, width=10.5 * cm, height=10.5 * cm, kind="proportional")
            gradcam_image.hAlign = "CENTER"
            story.append(Spacer(1, 0.08 * cm))
            story.append(gradcam_image)
            story.append(Spacer(1, 0.15 * cm))
        except Exception:
            story.append(Paragraph("GradCAM visualization is not available for this report.", styles["body"]))
    else:
        story.append(Paragraph("GradCAM visualization is not available for this report.", styles["body"]))
    story.append(
        Paragraph(
            "GradCAM visualization highlights the regions in the MRI scan that contributed most strongly to the model's prediction.",
            styles["caption"],
        )
    )

    # Cognitive Health Evaluation
    add_section_heading("Cognitive Health Evaluation")
    story.append(Paragraph(f"- Risk Level: {risk_label}", styles["bullet"]))
    story.append(Paragraph(f"- Risk Score: {risk_score_text}", styles["bullet"]))

    # AI Diagnostic Summary
    add_section_heading("AI Diagnostic Summary")
    story.append(Paragraph(safe_summary, styles["body"]))

    # Footer
    story.append(Spacer(1, 0.6 * cm))
    story.append(HRFlowable(width="100%", thickness=0.7, color=divider_color, spaceBefore=2, spaceAfter=6))
    story.append(Paragraph("Generated by Neuro Vision AI Platform", styles["footer_heading"]))
    story.append(
        Paragraph(
            "This report is generated using artificial intelligence and is intended to support clinical decision-making. "
            "It should not replace professional medical diagnosis.",
            styles["footer_text"],
        )
    )
    story.append(Paragraph(f"Generated on {generated_at.strftime('%B %d, %Y at %H:%M:%S')}", styles["footer_text"]))

    doc.build(story)
    return buffer.getvalue()


@app.post("/generate-report")
async def generate_report_pdf(payload: ReportRequest) -> StreamingResponse:
    confidence = float(payload.tumor_confidence if payload.tumor_confidence is not None else payload.confidence)
    risk       = float(payload.risk_score if payload.risk_score is not None else (payload.risk_probability or 0.0))
    summary    = payload.summary or payload.report_summary or ""
    
    # Extract base64 image data (e.g., "data:image/png;base64,iVBORw0KGgo..." -> "iVBORw0KGgo...")
    gradcam_base64: str | None = None
    if payload.gradcam_image:
        if payload.gradcam_image.startswith("data:image/png;base64,"):
            gradcam_base64 = payload.gradcam_image.replace("data:image/png;base64,", "")
        else:
            # Fallback for non-base64 image strings (legacy support)
            gradcam_base64 = payload.gradcam_image

    try:
        pdf_bytes = _build_pdf_report(
            tumor_prediction=payload.tumor_prediction,
            confidence=confidence,
            cognitive_risk=payload.cognitive_risk,
            risk_score=risk,
            report_summary=summary,
            gradcam_base64=gradcam_base64,
        )
    except Exception as error:
        logger.exception("%s PDF report generation failed", PLATFORM_NAME)
        raise HTTPException(status_code=500, detail="PDF report generation failed") from error

    logger.info("%s PDF report generated successfully", PLATFORM_NAME)
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="neurovision_report.pdf"'},
    )