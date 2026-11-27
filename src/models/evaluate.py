"""Evaluation metrics for binary machine-failure classifiers."""

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model, X, y) -> tuple[dict[str, float], list[list[int]]]:
    """Return positive-class metrics and a 0/1 confusion matrix."""
    predictions = model.predict(X)
    failure_scores = model.predict_proba(X)[:, 1]

    metrics = {
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1_score": float(f1_score(y, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y, failure_scores)),
    }
    matrix = confusion_matrix(y, predictions, labels=[0, 1]).tolist()
    return metrics, matrix
