"""Baseline classifiers for the AI4I 2020 dataset."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42


def create_models() -> dict:
    """Create baseline models with class weighting for the imbalanced target."""
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def train_model(model, X_train, y_train):
    """Fit one classifier on the supplied training data."""
    model.fit(X_train, y_train)
    return model
