from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("IBM-HR-Employee-Attrition-Performance.xlsx")
OUTPUT_DIR = Path("artifacts")
TARGET_COL = "Attrition"
EXCLUDED_FEATURES = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
N_SPLITS = 5
N_REPEATS = 10
BASE_SEED = 42


def map_feature_to_raw_column(transformed_name: str) -> str:
    if transformed_name.startswith("num__"):
        return transformed_name.replace("num__", "", 1)
    if transformed_name.startswith("cat__"):
        return transformed_name.replace("cat__", "", 1).split("_", 1)[0]
    return transformed_name


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str], random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def aggregated_feature_importance(model: Pipeline) -> pd.DataFrame:
    transformed_feature_names = model.named_steps["preprocessor"].get_feature_names_out().tolist()
    transformed_feature_scores = model.named_steps["classifier"].feature_importances_
    transformed_importance = pd.DataFrame(
        {
            "transformed_feature": transformed_feature_names,
            "importance": transformed_feature_scores,
        }
    )
    transformed_importance["raw_feature"] = transformed_importance["transformed_feature"].map(
        map_feature_to_raw_column
    )

    return (
        transformed_importance.groupby("raw_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def write_executive_summary(
    sample_size: int,
    attrition_rate: float,
    metrics_summary: dict[str, float],
    top_3: pd.DataFrame,
    top_10: pd.DataFrame,
) -> None:
    lines = [
        "# Executive Summary: Employee Attrition Risk Model",
        "",
        f"- Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Sample size: {sample_size} employees",
        f"- Attrition prevalence: {attrition_rate:.1%}",
        f"- Validation design: Repeated Stratified {N_SPLITS}-Fold CV ({N_REPEATS} repeats, {N_SPLITS * N_REPEATS} total folds)",
        "",
        "## Model Performance",
        f"- ROC-AUC (mean): {metrics_summary['roc_auc_mean']:.3f} (std: {metrics_summary['roc_auc_std']:.3f})",
        f"- Accuracy (mean): {metrics_summary['accuracy_mean']:.3f}",
        f"- Precision (mean): {metrics_summary['precision_mean']:.3f}",
        f"- Recall (mean): {metrics_summary['recall_mean']:.3f}",
        f"- F1 (mean): {metrics_summary['f1_mean']:.3f}",
        "",
        "## Stable Top 3 Drivers of Turnover",
        "Drivers are ranked by how often they appear in the top 3 across folds (stability), then by average importance.",
        "",
    ]

    for row in top_3.itertuples(index=False):
        lines.append(
            f"- {row.rank}. {row.raw_feature}: in top 3 for {row.top_3_frequency:.1%} of folds; avg importance {row.mean_importance:.4f}"
        )

    lines.extend(
        [
            "",
            "## Interpretation Guidance",
            "- These are predictive drivers, not causal proof.",
            "- Use drivers to prioritize policy review and targeted retention actions.",
            "- Re-run quarterly to monitor shifts in workforce dynamics.",
            "",
            "## Top 10 Stability Table",
            "",
            "| Rank | Feature | Top-3 Frequency | Mean Importance | Std Importance |",
            "|---|---|---:|---:|---:|",
        ]
    )

    for row in top_10.itertuples(index=False):
        lines.append(
            f"| {row.rank} | {row.raw_feature} | {row.top_3_frequency:.1%} | {row.mean_importance:.4f} | {row.std_importance:.4f} |"
        )

    (OUTPUT_DIR / "executive_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(DATA_PATH)
    y = (df[TARGET_COL] == "Yes").astype(int)
    X = df.drop(columns=[TARGET_COL] + EXCLUDED_FEATURES)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    metrics_rows: list[dict[str, float | int]] = []
    feature_rows: list[pd.DataFrame] = []

    for repeat_idx in range(N_REPEATS):
        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=BASE_SEED + repeat_idx,
        )
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model = build_pipeline(
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                random_state=BASE_SEED + repeat_idx * 100 + fold_idx,
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics_rows.append(
                {
                    "repeat": repeat_idx + 1,
                    "fold": fold_idx,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "f1": f1_score(y_test, y_pred, zero_division=0),
                    "roc_auc": roc_auc_score(y_test, y_proba),
                }
            )

            fold_importance = aggregated_feature_importance(model)
            fold_importance["repeat"] = repeat_idx + 1
            fold_importance["fold"] = fold_idx
            fold_importance["rank"] = fold_importance["importance"].rank(
                ascending=False, method="min"
            )
            fold_importance["in_top_3"] = fold_importance["rank"] <= 3
            feature_rows.append(fold_importance)

    metrics_df = pd.DataFrame(metrics_rows)
    all_feature_df = pd.concat(feature_rows, ignore_index=True)

    metrics_summary = {
        "accuracy_mean": float(metrics_df["accuracy"].mean()),
        "accuracy_std": float(metrics_df["accuracy"].std(ddof=1)),
        "precision_mean": float(metrics_df["precision"].mean()),
        "precision_std": float(metrics_df["precision"].std(ddof=1)),
        "recall_mean": float(metrics_df["recall"].mean()),
        "recall_std": float(metrics_df["recall"].std(ddof=1)),
        "f1_mean": float(metrics_df["f1"].mean()),
        "f1_std": float(metrics_df["f1"].std(ddof=1)),
        "roc_auc_mean": float(metrics_df["roc_auc"].mean()),
        "roc_auc_std": float(metrics_df["roc_auc"].std(ddof=1)),
        "num_folds": int(N_SPLITS * N_REPEATS),
    }

    stability_df = (
        all_feature_df.groupby("raw_feature", as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            mean_rank=("rank", "mean"),
            top_3_frequency=("in_top_3", "mean"),
        )
        .sort_values(["top_3_frequency", "mean_importance"], ascending=[False, False])
        .reset_index(drop=True)
    )
    stability_df.insert(0, "rank", range(1, len(stability_df) + 1))

    top_3 = stability_df.head(3).copy()
    top_10 = stability_df.head(10).copy()

    final_model = build_pipeline(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        random_state=BASE_SEED,
    )
    final_model.fit(X, y)

    joblib.dump(final_model, OUTPUT_DIR / "attrition_model.joblib")
    metrics_df.to_csv(OUTPUT_DIR / "cv_metrics.csv", index=False)
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    all_feature_df.to_csv(OUTPUT_DIR / "feature_importance_folds.csv", index=False)
    stability_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    top_3.to_csv(OUTPUT_DIR / "top_3_drivers.csv", index=False)
    write_executive_summary(
        sample_size=len(df),
        attrition_rate=float(y.mean()),
        metrics_summary=metrics_summary,
        top_3=top_3,
        top_10=top_10,
    )

    print("Model training and stability analysis complete.")
    print("Saved artifacts to:", OUTPUT_DIR.resolve())
    print("\nCross-validated metrics (mean +/- std):")
    print(f"- ROC-AUC: {metrics_summary['roc_auc_mean']:.4f} +/- {metrics_summary['roc_auc_std']:.4f}")
    print(f"- Accuracy: {metrics_summary['accuracy_mean']:.4f} +/- {metrics_summary['accuracy_std']:.4f}")
    print(f"- Precision: {metrics_summary['precision_mean']:.4f} +/- {metrics_summary['precision_std']:.4f}")
    print(f"- Recall: {metrics_summary['recall_mean']:.4f} +/- {metrics_summary['recall_std']:.4f}")
    print(f"- F1: {metrics_summary['f1_mean']:.4f} +/- {metrics_summary['f1_std']:.4f}")
    print("\nStable top 3 drivers (by top-3 frequency across folds):")
    for row in top_3.itertuples(index=False):
        print(
            f"{row.rank}. {row.raw_feature} (top-3 frequency={row.top_3_frequency:.2%}, "
            f"mean importance={row.mean_importance:.6f})"
        )


if __name__ == "__main__":
    main()
