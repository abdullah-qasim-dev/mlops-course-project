
"""Selected hyperparameters for the final Random Forest model."""

RANDOM_STATE = 42

BEST_MODEL_PARAMETERS = {
    "n_estimators": 400,
    "min_samples_split": 5,
    "min_samples_leaf": 1,
    "max_samples": 0.7,
    "max_features": None,
    "max_depth": 20,
    "criterion": "log_loss",
    "class_weight": {0: 1, 1: 5},
    "bootstrap": True,
    "random_state": RANDOM_STATE,
    "n_jobs": 2,
}
