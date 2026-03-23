"""
Install dependencies:
pip install pandas numpy scikit-learn joblib

Run with:
python training/train_cognitive_model.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_DATA_PATH = PROJECT_ROOT / "Data" / "alzheimers_disease_data.csv"
FALLBACK_DATA_PATH = PROJECT_ROOT / "Data" / "Dataset" / "alzheimers_disease_data.csv"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "cognitive_risk_model.pkl"
FEATURES_OUTPUT_PATH = PROJECT_ROOT / "models" / "cognitive_features.pkl"
MODEL_FEATURE_COLUMNS = ["Age", "EducationLevel", "MMSE", "FunctionalAssessment"]

TARGET_ALIASES = ["cognitive_risk", "Diagnosis"]


def resolve_data_path() -> Path:
    if PRIMARY_DATA_PATH.is_file():
        return PRIMARY_DATA_PATH
    if FALLBACK_DATA_PATH.is_file():
        return FALLBACK_DATA_PATH

    raise FileNotFoundError(
        "Dataset file not found. Checked: "
        f"{PRIMARY_DATA_PATH} and {FALLBACK_DATA_PATH}"
    )


def resolve_target_column(dataframe: pd.DataFrame) -> str:
    column_lookup = {column.lower(): column for column in dataframe.columns}

    for alias in TARGET_ALIASES:
        if alias in dataframe.columns:
            return alias
        alias_lower = alias.lower()
        if alias_lower in column_lookup:
            return column_lookup[alias_lower]

    raise ValueError(
        "Target column not found. Expected one of: "
        f"{TARGET_ALIASES}. Available columns: {list(dataframe.columns)}"
    )


def build_preprocessor(feature_frame: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = feature_frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [
        column for column in feature_frame.columns if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        transformers.append(("num", StandardScaler(), numeric_columns))

    if categorical_columns:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", encoder, categorical_columns))

    if not transformers:
        raise ValueError("No usable feature columns found after preprocessing.")

    return ColumnTransformer(transformers=transformers), numeric_columns, categorical_columns


def main() -> None:
    data_path = resolve_data_path()
    dataframe = pd.read_csv(data_path)
    print(f"Loaded dataset: {data_path}")
    print(f"Original dataset shape: {dataframe.shape}")

    target_column = resolve_target_column(dataframe)
    print(f"Using target column: {target_column}")

    required_columns = MODEL_FEATURE_COLUMNS + [target_column]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required compact-model columns: "
            f"{missing_columns}"
        )

    dataframe = dataframe.dropna(subset=required_columns).reset_index(drop=True)
    print(f"Dataset shape after dropping missing values: {dataframe.shape}")
    print(f"Training compact model with features: {MODEL_FEATURE_COLUMNS}")

    features = dataframe[MODEL_FEATURE_COLUMNS].copy()

    target = pd.to_numeric(dataframe[target_column], errors="coerce")
    valid_rows = target.notna()
    if not valid_rows.all():
        dropped_rows = int((~valid_rows).sum())
        print(f"Dropped {dropped_rows} rows with invalid target values.")
        features = features.loc[valid_rows].reset_index(drop=True)
        target = target.loc[valid_rows].reset_index(drop=True)

    target = target.astype(int)
    unique_targets = sorted(target.unique().tolist())
    if not set(unique_targets).issubset({0, 1}):
        raise ValueError(
            "Expected binary cognitive risk labels 0/1, found: "
            f"{unique_targets}"
        )

    preprocessor, numeric_columns, categorical_columns = build_preprocessor(features)
    print(f"Numeric features: {len(numeric_columns)}")
    print(f"Categorical features: {len(categorical_columns)}")

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42,
                ),
            ),
        ]
    )

    stratify_target = target if target.nunique() > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )

    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples: {len(x_test)}")

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    matrix = confusion_matrix(y_test, predictions)

    print("\nEvaluation Metrics")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    if probabilities is not None:
        auc = roc_auc_score(y_test, probabilities)
        print(f"ROC-AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(matrix)

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"Saved model to: {MODEL_OUTPUT_PATH}")

    joblib.dump(MODEL_FEATURE_COLUMNS, FEATURES_OUTPUT_PATH)
    print(f"Saved feature names to: {FEATURES_OUTPUT_PATH}")


if __name__ == "__main__":
    main()