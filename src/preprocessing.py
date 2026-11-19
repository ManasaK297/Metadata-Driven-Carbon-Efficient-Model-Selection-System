"""Data preprocessing utilities for dataset preparation."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_dataset(
    df: pd.DataFrame,
    target_col: str | None = None,
    test_size: float = 0.2,
    scale_features: bool = True,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series | None, pd.Series | None]:
    """Preprocess dataset: handle missing, encode, scale, split."""

    df = df.copy()

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(
                df[col].mode()[0] if not df[col].mode().empty else "unknown"
            )

    y = None
    if target_col is not None and target_col in df.columns:
        y = df[target_col]
        df = df.drop(columns=[target_col])

    for col in categorical_cols:
        if col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Always encode target labels to numeric to ensure compatibility
    # with all estimators (e.g., XGBoost expects numeric labels)
    y_encoded = None
    if y is not None:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))

    stratify_col = y_encoded if (stratify and y_encoded is not None) else None

    if y_encoded is not None:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                df, y_encoded, test_size=test_size, random_state=42, stratify=stratify_col
            )
        except ValueError:
            # Fallback for very small datasets or when stratification is impossible
            X_train, X_test, y_train, y_test = train_test_split(
                df, y_encoded, test_size=test_size, random_state=42, stratify=None
            )
    else:
        X_train, X_test = train_test_split(
            df, test_size=test_size, random_state=42
        )
        y_train, y_test = None, None

    if scale_features and len(numeric_cols) > 0:
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        numeric_features = [col for col in numeric_cols if col in X_train.columns]
        if numeric_features:
            X_train_scaled[numeric_features] = scaler.fit_transform(
                X_train[numeric_features]
            )
            X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

        X_train = X_train_scaled
        X_test = X_test_scaled

    if y_encoded is not None:
        return X_train, X_test, pd.Series(y_train), pd.Series(y_test)
    else:
        return X_train, X_test, None, None


if __name__ == "__main__":
    import tempfile

    sample_data = pd.DataFrame(
        {
            "age": [25, 30, np.nan, 45, 50],
            "job": ["A", "B", "A", np.nan, "C"],
            "score": [100, 200, 150, 300, np.nan],
            "target": [0, 1, 0, 1, 1],
        }
    )

    print("Original data:")
    print(sample_data)
    print()

    X_train, X_test, y_train, y_test = preprocess_dataset(sample_data, target_col="target")
    print("Train features shape:", X_train.shape)
    print("Test features shape:", X_test.shape)
    print("Train target shape:", y_train.shape)
    print("Test target shape:", y_test.shape)
