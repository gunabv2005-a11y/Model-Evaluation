# Model Evaluation Toolkit (ML-4)

This project compares two binary classifiers on an intentionally imbalanced dataset and shows why accuracy should not be the only metric used to judge a model.

## What it includes
- Reproducible imbalanced dataset (92% / 8% target weights) generated with `make_classification`.
- Stratified 5-fold cross-validation with mean and standard deviation for accuracy, F1, precision and recall.
- Held-out test set metrics for Logistic Regression and Random Forest.
- Confusion matrix plot for each model.
- ROC curves with AUC values on one plot.
- Precision-recall curves with the positive-class prevalence baseline.
- CSV tables and a text summary written into `outputs/`.

## Setup (Windows / PowerShell)
Run these commands from this project folder:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python model_evaluation.py
```

If PowerShell blocks environment activation, you can still install and run using the venv's Python directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe model_evaluation.py
```

## Output files
The program creates an `outputs` directory containing:
- `cross_validation_results.csv`
- `test_set_results.csv`
- `confusion_matrix_logistic_regression.png`
- `confusion_matrix_random_forest.png`
- `roc_curves.png`
- `precision_recall_curves.png`
- `evaluation_summary.txt`

## Evaluation notes
- **Accuracy** is the fraction of all predictions that are correct. With imbalanced labels, a model can achieve high accuracy while missing most minority examples.
- **Precision** answers: of predicted positive cases, how many were truly positive?
- **Recall** answers: of actual positive cases, how many did the model find?
- **F1** balances precision and recall using their harmonic mean.
- **ROC-AUC** evaluates how well scores rank positives above negatives across thresholds.
- **Precision-recall** is particularly informative when the positive class is rare because it focuses on positive-class detection and precision rather than the large number of true negatives.
- **Confusion matrix** shows true negatives, false positives, false negatives and true positives.

## Model comparison and deployment choice
Use the generated test metrics and curves to make the deployment decision. If the use case requires finding as many positive cases as possible, prioritize recall while checking the false-positive burden. If false alarms are expensive, prioritize precision. F1 is a useful compromise, but a deployment threshold should be chosen based on real-world costs. The code reports the model with the highest test-set F1 as a starting point, not an automatic final deployment decision.

## Dataset caveat
This task uses a synthetic dataset for a reproducible demonstration. It is not evidence that either model is ready for a real production system. For a stronger portfolio project, repeat the evaluation with a documented real-world imbalanced dataset and explain its limitations.
