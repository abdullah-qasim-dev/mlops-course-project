
"""Hyperparameter tuning for the Random Forest model."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data.preprocessing import create_preprocessor


RANDOM_STATE = 42
TUNING_ITERATIONS = 40
CV_FOLDS = 5


def create_tuning_search():
    """Create a RandomizedSearchCV object for Random Forest."""

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        (
            "classifier",
            RandomForestClassifier(
                bootstrap=True,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
    ])

    parameter_distributions = {
        "classifier__n_estimators": [200, 300, 400, 500],
        "classifier__max_depth": [None, 8, 12, 16, 20, 30],
        "classifier__min_samples_split": [2, 5, 10, 15, 20],
        "classifier__min_samples_leaf": [1, 2, 3, 4, 6, 8],
        "classifier__max_features": ["sqrt", "log2", None],
        "classifier__class_weight": [
            "balanced",
            "balanced_subsample",
            {0: 1, 1: 5},
            {0: 1, 1: 10},
            {0: 1, 1: 15},
        ],
        "classifier__max_samples": [0.7, 0.85, 1.0],
        "classifier__criterion": ["gini", "entropy", "log_loss"],
    }

    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_distributions,
        n_iter=TUNING_ITERATIONS,
        scoring="average_precision",
        cv=cross_validation,
        random_state=RANDOM_STATE,
        n_jobs=2,
        verbose=2,
        refit=True,
        return_train_score=True,
    )

    return search
