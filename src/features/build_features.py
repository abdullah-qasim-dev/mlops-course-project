"""Separate model input features from the prediction target."""

import pandas as pd

from src.data.validation import MODEL_FEATURE_COLUMNS, TARGET_COLUMN


def build_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return sensor measurements and machine type as X, and failure as y."""
    X = data[MODEL_FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].copy()
    return X, y
