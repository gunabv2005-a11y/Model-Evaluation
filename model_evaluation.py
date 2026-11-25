"""Model evaluation example for an imbalanced binary classification problem."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42



def build_dataset():
    """Create a reproducible dataset with a small minority class."""
    features, labels = make_classification(
        n_samples=6000,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        n_clusters_per_class=2,
        weights=[0.92, 0.08],
        flip_y=0.015,
        random_state=RANDOM_STATE,
    )
    return features, labels



def build_models():
    """Return two different classifiers using the same training data."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }



def run_cross_validation(models, x_train, y_train):
    """Use stratified folds so each fold keeps a similar class distribution."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "f1": "f1", "precision": "precision", "recall": "recall"}
    rows = []

    for name, model in models.items():
        scores = cross_validate(model, x_train, y_train, cv=folds, scoring=scoring, n_jobs=-1)
        row = {"Model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric.title()} Mean"] = values.mean()
            row[f"{metric.title()} Std"] = values.std()
        rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "cross_validation_results.csv", index=False)
    return results



def evaluate_on_test(models, x_train, x_test, y_train, y_test):
    """Fit each model once and save test metrics and evaluation plots."""
    test_rows = []
    roc_plot_data = []
    pr_plot_data = []
    predictions = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions[name] = predicted

        test_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predicted),
            "Precision": precision_score(y_test, predicted, zero_division=0),
            "Recall": recall_score(y_test, predicted, zero_division=0),
            "F1": f1_score(y_test, predicted, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
        })

        fpr, tpr, _ = roc_curve(y_test, probabilities)
        precision, recall, _ = precision_recall_curve(y_test, probabilities)
        roc_plot_data.append((name, fpr, tpr, roc_auc_score(y_test, probabilities)))
        pr_plot_data.append((name, recall, precision))

        cm_display = ConfusionMatrixDisplay.from_predictions(
            y_test, predicted, display_labels=["Class 0", "Class 1"],
            values_format="d", cmap="Blues", colorbar=False
        )
        cm_display.ax_.set_title(f"Confusion Matrix - {name}")
        cm_display.figure_.tight_layout()
        cm_display.figure_.savefig(OUTPUT_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=160)
        plt.close(cm_display.figure_)

    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(OUTPUT_DIR / "test_set_results.csv", index=False)


    # ROC curves for both classifiers on the same chart
    plt.figure(figsize=(8, 6))
    for name, fpr, tpr, auc_value in roc_plot_data:
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_value:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=160)
    plt.close()

    # Precision-recall curves for both classifiers
    plt.figure(figsize=(8, 6))
    baseline = y_test.mean()
    for name, recall, precision in pr_plot_data:
        plt.plot(recall, precision, label=name)
    plt.axhline(baseline, linestyle="--", label=f"Positive-class baseline = {baseline:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="best")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "precision_recall_curves.png", dpi=160)
    plt.close()

    return test_results, predictions



def write_summary(cv_results, test_results, class_counts):
    """Write a short plain-text report with the actual run's results."""
    best_model = test_results.sort_values("F1", ascending=False).iloc[0]
    report_lines = [
        "MODEL EVALUATION SUMMARY",
        "=======================",
        f"Class distribution (whole dataset): {class_counts.to_dict()}",
        "",
        "Cross-validation (5 stratified folds):",
        cv_results.to_string(index=False),
        "",
        "Held-out test-set results:",
        test_results.to_string(index=False),
        "",
        f"Model with the highest test F1 in this run: {best_model['Model']} (F1={best_model['F1']:.4f}).",
        "Deployment decision should consider the cost of false negatives and false positives, not only accuracy.",
        "For an imbalanced dataset, precision-recall curves and recall/precision at a chosen threshold are often more useful than accuracy alone.",
    ]
    (OUTPUT_DIR / "evaluation_summary.txt").write_text("\n".join(report_lines), encoding="utf-8")



def main():
    x, y = build_dataset()
    print("Class counts:")
    print(pd.Series(y).value_counts().sort_index())
    print("\nClass proportions:")
    print(pd.Series(y).value_counts(normalize=True).sort_index().round(3))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    models = build_models()
    cv_results = run_cross_validation(models, x_train, y_train)
    test_results, _ = evaluate_on_test(models, x_train, x_test, y_train, y_test)
    write_summary(cv_results, test_results, pd.Series(y).value_counts().sort_index())



if __name__ == "__main__":
    main()
