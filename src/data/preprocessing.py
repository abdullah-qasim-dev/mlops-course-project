"""Training-only numerical scaling and categorical encoding."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.validation import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def create_preprocessor() -> ColumnTransformer:
    """Create the scaler and encoder used for the input features."""
    return ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), NUMERICAL_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def preprocess_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """Fit preprocessing on training data and apply it to both data splits."""
    preprocessor = create_preprocessor()
    train_values = preprocessor.fit_transform(X_train)
    test_values = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(
        train_values, columns=feature_names, index=X_train.index
    )
    X_test_processed = pd.DataFrame(
        test_values, columns=feature_names, index=X_test.index
    )
    return X_train_processed, X_test_processed, preprocessor
