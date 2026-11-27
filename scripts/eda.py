"""Exploratory analysis for the AI4I 2020 predictive maintenance dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "ai4i2020.csv"
REPORT_PATH = ROOT_DIR / "reports" / "eda_report.md"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

TARGET = "Machine failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
IDENTIFIER_COLUMNS = ["UDI", "Product ID"]


def save_target_distribution(data: pd.DataFrame) -> None:
    counts = data[TARGET].value_counts().reindex([0, 1], fill_value=0)
    percentages = counts / len(data) * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["No failure (0)", "Failure (1)"], counts.values, color=["#4C78A8", "#E45756"])
    ax.bar_label(
        bars,
        labels=[f"{count:,} ({percentage:.2f}%)" for count, percentage in zip(counts, percentages)],
        padding=4,
    )
    ax.set_title("Machine Failure Distribution")
    ax.set_ylabel("Number of records")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "target_distribution.png", dpi=150)
    plt.close(fig)


def save_numerical_distributions(data: pd.DataFrame, features: list[str]) -> None:
    fig, axes = plt.subplots(len(features), 1, figsize=(9, 3 * len(features)))
    for ax, feature in zip(axes, features):
        sns.histplot(data=data, x=feature, kde=True, ax=ax)
        ax.set_title(f"Distribution of {feature}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "numerical_distributions.png", dpi=150)
    plt.close(fig)


def save_numerical_outliers(data: pd.DataFrame, features: list[str]) -> None:
    fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 5))
    for ax, feature in zip(axes, features):
        sns.boxplot(data=data, y=feature, ax=ax, color="#72B7B2")
        ax.set_title(feature)
    fig.suptitle("Numerical Feature Boxplots")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "numerical_outliers.png", dpi=150)
    plt.close(fig)


def save_machine_type_analysis(data: pd.DataFrame) -> None:
    type_counts = data["Type"].value_counts().sort_index()
    failure_rates = data.groupby("Type", observed=False)[TARGET].mean().reindex(type_counts.index) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x=type_counts.index, y=type_counts.values, ax=axes[0], color="#4C78A8")
    axes[0].set_title("Machine Type Counts")
    axes[0].set_xlabel("Machine type")
    axes[0].set_ylabel("Number of records")

    sns.barplot(x=failure_rates.index, y=failure_rates.values, ax=axes[1], color="#F58518")
    axes[1].set_title("Machine Failure Rate by Type")
    axes[1].set_xlabel("Machine type")
    axes[1].set_ylabel("Failure rate (%)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "machine_type_analysis.png", dpi=150)
    plt.close(fig)


def save_correlation_heatmap(data: pd.DataFrame, features: list[str]) -> None:
    correlation = data[features + [TARGET]].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def save_failure_relationships(data: pd.DataFrame, features: list[str]) -> None:
    fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 5))
    for ax, feature in zip(axes, features):
        sns.boxplot(data=data, x=TARGET, y=feature, ax=ax, showfliers=False)
        ax.set_title(feature)
        ax.set_xlabel("Machine failure (0/1)")
    fig.suptitle("Numerical Features by Machine Failure")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "features_by_failure.png", dpi=150)
    plt.close(fig)


def save_failure_mode_counts(data: pd.DataFrame) -> None:
    mode_counts = data[FAILURE_MODES].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(mode_counts.index, mode_counts.values, color="#E45756")
    ax.bar_label(bars, padding=3)
    ax.set_title("Failure Mode Counts")
    ax.set_ylabel("Number of records")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "failure_mode_counts.png", dpi=150)
    plt.close(fig)


def create_report(data: pd.DataFrame, numerical_features: list[str]) -> None:
    row_count, column_count = data.shape
    target_counts = data[TARGET].value_counts().reindex([0, 1], fill_value=0)
    failure_count = int(target_counts.loc[1])
    failure_rate = float(target_counts.loc[1] / row_count * 100)
    imbalance_ratio = (
        float(target_counts.loc[0] / target_counts.loc[1])
        if target_counts.loc[1]
        else float("inf")
    )

    missing_by_column = data.isna().sum()
    missing_columns = missing_by_column[missing_by_column > 0]
    numerical_summary = data[numerical_features].describe().round(2)

    quartiles = data[numerical_features].quantile([0.25, 0.75])
    first_quartile = quartiles.loc[0.25]
    third_quartile = quartiles.loc[0.75]
    spread = third_quartile - first_quartile
    lower_bounds = first_quartile - 1.5 * spread
    upper_bounds = third_quartile + 1.5 * spread
    outlier_counts = (
        (data[numerical_features] < lower_bounds)
        | (data[numerical_features] > upper_bounds)
    ).sum()
    outlier_summary = pd.DataFrame(
        {
            "Lower IQR bound": lower_bounds,
            "Upper IQR bound": upper_bounds,
            "Outlier count": outlier_counts,
            "Outlier percent": outlier_counts / row_count * 100,
        }
    ).round(2)

    failure_by_type = data.groupby("Type", observed=False)[TARGET].agg(
        failures="sum", records="count", failure_rate="mean"
    )
    failure_by_type["failure_rate"] *= 100
    failure_by_type = failure_by_type.round(2)

    feature_correlations = (
        data[numerical_features].corrwith(data[TARGET])
        .sort_values(key=lambda values: values.abs(), ascending=False)
        .round(3)
    )
    mode_counts = data[FAILURE_MODES].sum()
    failures_with_mode = data.loc[data[TARGET] == 1, FAILURE_MODES].sum()
    mode_summary = pd.DataFrame(
        {
            "Records with mode": mode_counts,
            "Percent of all records": mode_counts / row_count * 100,
            "Failed records with mode": failures_with_mode,
            "Percent of failures with mode": (
                failures_with_mode / failure_count * 100
                if failure_count
                else pd.Series(0.0, index=FAILURE_MODES)
            ),
        }
    ).round(2)

    strongest_correlated_feature = feature_correlations.abs().idxmax()
    most_outliers_feature = outlier_counts.idxmax()
    most_common_failure_mode = mode_counts.idxmax()
    highest_failure_type = failure_by_type["failure_rate"].idxmax()
    target_summary = pd.DataFrame(
        {
            "Count": target_counts,
            "Percent": (target_counts / row_count * 100).round(2),
        }
    )

    lines = [
        "# AI4I 2020 Exploratory Data Analysis",
        "",
        "## Key findings",
        "",
        f"- {failure_count:,} of {row_count:,} records show machine failure ({failure_rate:.2f}%); "
        f"the non-failure to failure ratio is {imbalance_ratio:.1f}:1.",
        f"- **{strongest_correlated_feature}** has the largest absolute Pearson correlation with "
        f"machine failure (r={feature_correlations.loc[strongest_correlated_feature]:.3f}); "
        "this is an association, not causation.",
        f"- **{most_outliers_feature}** has the most IQR outliers "
        f"({int(outlier_summary.loc[most_outliers_feature, 'Outlier count']):,}).",
        f"- **{most_common_failure_mode}** is the most frequently flagged failure mode "
        f"({int(mode_summary.loc[most_common_failure_mode, 'Records with mode']):,} records).",
        f"- Type **{highest_failure_type}** has the highest observed failure rate "
        f"({failure_by_type.loc[highest_failure_type, 'failure_rate']:.2f}%).",
        "",
        "## Dataset overview",
        "",
        f"- **Shape:** {row_count:,} rows x {column_count} columns",
        f"- **Columns:** {', '.join(data.columns)}",
        "",
        "### Data types",
        "",
        "```text",
        data.dtypes.astype(str).to_string(),
        "```",
        "",
        "### Sample records (first five rows)",
        "",
        "```text",
        data.head().to_string(index=False),
        "```",
        "",
        "## Data quality and descriptive statistics",
        "",
        f"- **Missing values:** {int(missing_by_column.sum()):,} total",
        "- **Missing values by column:**",
        "```text",
        missing_columns.to_string() if not missing_columns.empty else "No missing values.",
        "```",
        f"- **Exact duplicate rows:** {int(data.duplicated().sum()):,}",
        "",
        "### Descriptive statistics for numerical features",
        "",
        "```text",
        numerical_summary.transpose().to_string(),
        "```",
        "",
        "### IQR outlier counts",
        "",
        "Values outside 1.5 x IQR bounds are flagged for exploration; they are not removed.",
        "",
        "```text",
        outlier_summary.to_string(),
        "```",
        "",
        "## Machine failure target",
        "",
        "```text",
        target_summary.rename_axis(TARGET).to_string(),
        "```",
        f"- **Failure rate:** {failure_rate:.2f}%",
        f"- **Non-failure to failure ratio:** {imbalance_ratio:.1f}:1",
        "- The target is imbalanced, so accuracy alone may not describe future model performance well.",
        "",
        "![Machine failure distribution](figures/target_distribution.png)",
        "",
        "## Numerical features and relationships",
        "",
        "### Correlation with machine failure",
        "",
        "Pearson correlations are descriptive associations, not evidence of causation.",
        "",
        "```text",
        feature_correlations.rename("Pearson correlation").to_string(),
        "```",
        "",
        "![Numerical feature distributions](figures/numerical_distributions.png)",
        "",
        "![Numerical feature boxplots](figures/numerical_outliers.png)",
        "",
        "![Correlation heatmap](figures/correlation_heatmap.png)",
        "",
        "![Numerical features grouped by failure](figures/features_by_failure.png)",
        "",
        "## Machine type",
        "",
        "Failure counts and rates by machine type:",
        "",
        "```text",
        failure_by_type.to_string(),
        "```",
        "",
        "![Machine type counts and failure rates](figures/machine_type_analysis.png)",
        "",
        "## Failure modes",
        "",
        "Mode flags can overlap, so their counts do not necessarily add up to the number of failed machines.",
        "",
        "```text",
        mode_summary.to_string(),
        "```",
        "",
        "![Failure mode counts](figures/failure_mode_counts.png)",
        "",
        "## Scope",
        "",
        "This phase is exploratory only. The raw CSV was read without modification; no preprocessing, "
        "feature engineering, or model training was performed.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    required_columns = {"Type", TARGET, *FAILURE_MODES}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}")

    excluded_columns = {TARGET, *FAILURE_MODES}
    excluded_columns.update(IDENTIFIER_COLUMNS)
    # Keep identifiers and failure-mode labels separate from sensor measurements.
    numerical_features = [
        column
        for column in data.select_dtypes(include="number").columns
        if column not in excluded_columns
    ]
    if not numerical_features:
        raise ValueError("No numerical measurement features were found in the dataset.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    save_target_distribution(data)
    save_numerical_distributions(data, numerical_features)
    save_numerical_outliers(data, numerical_features)
    save_machine_type_analysis(data)
    save_correlation_heatmap(data, numerical_features)
    save_failure_relationships(data, numerical_features)
    save_failure_mode_counts(data)
    create_report(data, numerical_features)

    print(f"EDA report saved to: {REPORT_PATH}")
    print(f"EDA figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
