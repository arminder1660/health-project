"""CardiaNova - Cardiovascular Risk Model Training Pipeline

Trains and calibrates a cardiovascular disease prediction model on the
70,000-patient clinical dataset (UCI "Cardiovascular Disease" set).

Design notes:
- The model is trained by an EXPLICIT step (`python -m src.train`). It is never
  trained lazily inside an HTTP request, so a user never waits minutes for a
  first-request cold start. If the artifact is missing the API returns a clear
  error telling the operator to run the training step.
- Probabilities are calibrated (sigmoid / Platt scaling, 5-fold) because the
  downstream triage logic treats the score as a real risk percentage.
- A single held-out split is reported *and* a repeated stratified 5x5
  cross-validation with 95% confidence intervals, because one split cannot tell
  you whether a 0.4-point metric change is real or noise. Per-subgroup metrics
  and sensitivity/specificity at the shipped decision threshold are reported
  alongside accuracy, since a triage aid is judged on who it misses.

Features:
- Age (Years)
- Biological Sex
- Body Mass Index (BMI)
- Resting Systolic & Diastolic Blood Pressure
- Serum Cholesterol Level (Normal / Above Normal / High)
- Fasting Glucose Level (Normal / Pre-diabetic / Diabetic)
- Cigarette Smoking Status
- Alcohol Consumption
- Physical Activity Level

Pulse pressure (systolic - diastolic) was removed as a feature: it is a
deterministic function of two columns already in the set, and permutation
importance measured it at +0.0001 ROC-AUC, i.e. indistinguishable from noise.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import t as student_t

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cardio_train.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_DIR / "cardio_model.joblib"

FEATURE_NAMES = [
    "age_years",
    "gender",
    "bmi",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]

FEATURE_LABELS = {
    "age_years": "Patient Age (Years)",
    "gender": "Biological Sex (1=Female, 2=Male)",
    "bmi": "Body Mass Index (BMI, kg/m²)",
    "ap_hi": "Resting Systolic Blood Pressure (mm Hg)",
    "ap_lo": "Resting Diastolic Blood Pressure (mm Hg)",
    "cholesterol": "Serum Cholesterol (1=Normal, 2=Above, 3=High)",
    "gluc": "Fasting Glucose (1=Normal, 2=Pre-diabetic, 3=High)",
    "smoke": "Cigarette Smoking (0=No, 1=Yes)",
    "alco": "Alcohol Consumption (0=No, 1=Yes)",
    "active": "Physical Activity (0=No, 1=Yes)",
}

# The decision threshold the app actually ships with. `model.predict()` uses the
# same 0.5 internally; naming it here keeps the reported sensitivity and
# specificity tied to the operating point the patient really experiences.
OPERATING_THRESHOLD = 0.5

# Repeated stratified k-fold: 5 folds x 5 repeats = 25 independent estimates of
# each metric, which is enough to publish a confidence interval instead of a
# single number that cannot be distinguished from noise.
CV_SPLITS = 5
CV_REPEATS = 5

# Age bands for the per-subgroup report. Chosen to give every band a few thousand
# patients on the held-out split while separating the clinically distinct groups.
AGE_BANDS = [(0.0, 45.0, "under 45"), (45.0, 55.0, "45-54"), (55.0, 200.0, "55 and over")]

# Number of test rows used for the (relatively expensive) permutation importance pass.
IMPORTANCE_SAMPLE_SIZE = 3000
IMPORTANCE_REPEATS = 5
# Number of points kept for the persisted ROC curve (keeps the artifact small).
ROC_CURVE_POINTS = 60


class ModelNotTrainedError(RuntimeError):
    """Raised when inference is attempted before the model artifact exists."""


def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    print(f"Loading raw data from: {csv_path}")
    df = pd.read_csv(csv_path, sep=";")
    initial_count = len(df)
    print(f"Initial record count: {initial_count:,}")

    # Convert age from days to years
    df["age_years"] = df["age"] / 365.25

    # Calculate Body Mass Index: weight(kg) / (height(m))^2
    height_m = df["height"] / 100.0
    df["bmi"] = df["weight"] / (height_m ** 2)

    # Calculate pulse pressure (systolic - diastolic)
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

    # Filter physiological outliers
    clean_mask = (
        (df["ap_hi"] >= 70) & (df["ap_hi"] <= 240) &
        (df["ap_lo"] >= 40) & (df["ap_lo"] <= 150) &
        (df["ap_hi"] > df["ap_lo"]) &
        (df["height"] >= 120) & (df["height"] <= 220) &
        (df["weight"] >= 35) & (df["weight"] <= 220) &
        (df["bmi"] >= 13.0) & (df["bmi"] <= 60.0)
    )
    df_clean = df[clean_mask].copy()
    cleaned_count = len(df_clean)
    print(f"Cleaned record count: {cleaned_count:,} (Removed {initial_count - cleaned_count:,} invalid/outlier records)")

    # Duplicate patients hide behind the per-row `id` column: two records for
    # the same clinical profile look distinct if you deduplicate on every
    # column. Deduplicate on the clinical columns instead, and do it *before*
    # the split so twins cannot land on both sides of it.
    # Measured on this file: 24 of 70,000 rows, so this is correctness hygiene
    # rather than a material score change -- but the leak would still be
    # wrong, and it would matter on a dataset with heavier duplication.
    clinical_cols = [c for c in df_clean.columns if c != "id"]
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=clinical_cols).reset_index(drop=True)
    print(
        f"Removed {before_dedup - len(df_clean):,} duplicate patients "
        f"-> {len(df_clean):,} unique patients remain"
    )

    return df_clean


def _downsample_curve(xs, ys, points: int = ROC_CURVE_POINTS):
    """Evenly thins a curve down to at most `points` pairs for compact storage."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) <= points:
        idx = np.arange(len(xs))
    else:
        idx = np.unique(np.linspace(0, len(xs) - 1, points).astype(int))
    return [
        {"x": round(float(xs[i]), 4), "y": round(float(ys[i]), 4)} for i in idx
    ]


def _new_base_model() -> HistGradientBoostingClassifier:
    """The untrained estimator.

    Training and cross-validation both build the model through here so the
    reported cross-validated numbers describe the same estimator that ships.
    """
    return HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42,
    )


def _mean_and_ci(values) -> Dict[str, float]:
    """Mean and 95% confidence interval for a set of per-fold scores.

    Uses the t-distribution rather than the normal approximation: 25 folds is a
    small sample, and quoting 1.96 standard errors would publish an interval
    that is narrower than the data supports.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    half_width = float(student_t.ppf(0.975, n - 1) * arr.std(ddof=1) / np.sqrt(n))
    return {
        "mean": mean,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "std": float(arr.std(ddof=1)),
    }


def operating_point_metrics(
    y_true, y_prob, threshold: float = OPERATING_THRESHOLD
) -> Dict[str, Any]:
    """Sensitivity, specificity, PPV and NPV at the threshold the app ships with.

    Accuracy alone cannot describe a screening aid. The outcome is close to a
    coin flip in this dataset, so a model that answers "no disease" for everyone
    already scores about 50%. The numbers that decide whether the tool is safe
    are how many cases it catches (sensitivity) and how many of its alarms are
    real (PPV).
    """
    predicted = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

    def rate(numerator: int, denominator: int):
        return round(100.0 * numerator / denominator, 2) if denominator else None

    return {
        "threshold": threshold,
        "sensitivity": rate(tp, tp + fn),
        "specificity": rate(tn, tn + fp),
        "ppv": rate(tp, tp + fp),
        "npv": rate(tn, tn + fn),
        "prevalence": rate(tp + fn, tp + fp + fn + tn),
        "confusion": {
            "true_positive": int(tp),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_negative": int(tn),
        },
    }


def subgroup_metrics(X_raw: pd.DataFrame, y_true, y_prob) -> list:
    """The same metrics broken down by age band and by sex.

    One global score hides who the model fails. If the model leans on blood
    pressure, it will look strong on older patients and weak on younger ones,
    and the aggregate number will never show that.
    """
    groups = [
        ("Age band", label, (X_raw["age_years"] >= low) & (X_raw["age_years"] < high))
        for low, high, label in AGE_BANDS
    ]
    groups.append(("Sex", "Female", X_raw["gender"] == 1))
    groups.append(("Sex", "Male", X_raw["gender"] == 2))

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    report = []
    for dimension, label, mask in groups:
        members = mask.to_numpy()
        y_group, prob_group = y_true[members], y_prob[members]

        entry: Dict[str, Any] = {
            "dimension": dimension,
            "group": label,
            "n": int(members.sum()),
        }
        # A subgroup containing only one outcome class has no defined AUC.
        entry["roc_auc"] = (
            round(float(roc_auc_score(y_group, prob_group)) * 100, 2)
            if len(y_group) and len(np.unique(y_group)) == 2
            else None
        )
        metrics = operating_point_metrics(y_group, prob_group)
        metrics.pop("confusion")  # the aggregate confusion matrix is reported once
        entry.update(metrics)
        report.append(entry)
    return report


def repeated_cross_validation(X: pd.DataFrame, y) -> Dict[str, Any]:
    """Cross-validates the deployed pipeline, scaling and calibration included.

    Run on the training split only. Cross-validating over the whole dataset would
    let the folds see the held-out rows, and keeping those rows unseen is the
    entire reason a held-out split exists.
    """
    cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=42)
    total = CV_SPLITS * CV_REPEATS
    accuracy, roc_auc, brier = [], [], []

    print(f"\nRepeated stratified {CV_SPLITS}x{CV_REPEATS} cross-validation ({total} fits)...")
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        scaler = StandardScaler()
        X_fold_train = scaler.fit_transform(X.iloc[train_idx])
        X_fold_test = scaler.transform(X.iloc[test_idx])
        y_fold_train, y_fold_test = y.iloc[train_idx], y.iloc[test_idx]

        model = CalibratedClassifierCV(estimator=_new_base_model(), method="sigmoid", cv=5)
        model.fit(X_fold_train, y_fold_train)
        prob = model.predict_proba(X_fold_test)[:, 1]

        accuracy.append(
            accuracy_score(y_fold_test, (prob >= OPERATING_THRESHOLD).astype(int)) * 100
        )
        roc_auc.append(roc_auc_score(y_fold_test, prob) * 100)
        brier.append(brier_score_loss(y_fold_test, prob))
        print(f"  fold {fold:2d}/{total}  roc_auc {roc_auc[-1]:.2f}%  brier {brier[-1]:.4f}")

    report: Dict[str, Any] = {
        "scheme": f"RepeatedStratifiedKFold(n_splits={CV_SPLITS}, n_repeats={CV_REPEATS})",
        "folds": total,
    }
    for name, values, digits in (
        ("accuracy", accuracy, 2),
        ("roc_auc", roc_auc, 2),
        ("brier_score", brier, 4),
    ):
        report[name] = {
            key: round(value, digits) for key, value in _mean_and_ci(values).items()
        }
    return report


def _build_diagnostics(calibrated_model, X_test_scaled, y_test, y_prob):
    """Computes permutation feature importance, the ROC curve and the calibration curve."""
    # --- Permutation feature importance (subsampled for runtime) ---
    sample_n = min(IMPORTANCE_SAMPLE_SIZE, len(X_test_scaled))
    X_sample = X_test_scaled[:sample_n]
    y_sample = y_test.iloc[:sample_n]
    perm = permutation_importance(
        calibrated_model,
        X_sample,
        y_sample,
        n_repeats=IMPORTANCE_REPEATS,
        random_state=42,
        scoring="roc_auc",
    )
    feature_importance = sorted(
        [
            {
                "feature": name,
                "label": FEATURE_LABELS.get(name, name),
                "importance": round(float(mean), 4),
                "std": round(float(std), 4),
            }
            for name, mean, std in zip(FEATURE_NAMES, perm.importances_mean, perm.importances_std)
        ],
        key=lambda d: d["importance"],
        reverse=True,
    )

    # --- ROC curve (downsampled) ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_points = _downsample_curve(fpr, tpr)

    # --- Calibration curve / reliability diagram ---
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")
    calibration = [
        {"predicted": round(float(p), 4), "observed": round(float(t), 4)}
        for p, t in zip(prob_pred, prob_true)
    ]

    return feature_importance, roc_points, calibration


def train_model(force: bool = False):
    if MODEL_SAVE_PATH.exists() and not force:
        print(f"Model already exists at: {MODEL_SAVE_PATH}")
        return joblib.load(MODEL_SAVE_PATH)

    df = load_and_clean_data(DATA_PATH)

    X = df[FEATURE_NAMES]
    y = df["cardio"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train):,} | Test set: {len(X_test):,}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a fast, high-accuracy Gradient Boosting classifier with probability calibration
    base_model = _new_base_model()

    # 5-fold cross-validated probability calibration (sigmoid / Platt scaling)
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=5)
    calibrated_model.fit(X_train_scaled, y_train)

    # Evaluation
    y_pred = calibrated_model.predict(X_test_scaled)
    y_prob = calibrated_model.predict_proba(X_test_scaled)[:, 1]

    accuracy = float(accuracy_score(y_test, y_pred) * 100)
    roc_auc = float(roc_auc_score(y_test, y_prob) * 100)
    brier = float(brier_score_loss(y_test, y_prob))

    # --- Baseline comparison: uncalibrated logistic regression ---
    print("\nFitting logistic-regression baseline for comparison...")
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    baseline_model.fit(X_train_scaled, y_train)
    base_prob = baseline_model.predict_proba(X_test_scaled)[:, 1]
    baseline_metrics = {
        "name": "Logistic Regression (baseline)",
        "accuracy": round(float(accuracy_score(y_test, baseline_model.predict(X_test_scaled)) * 100), 2),
        "roc_auc": round(float(roc_auc_score(y_test, base_prob) * 100), 2),
        "brier_score": round(float(brier_score_loss(y_test, base_prob)), 4),
    }

    # --- Diagnostics: permutation importance, ROC curve, calibration curve ---
    print("Computing permutation importance, ROC and calibration curves...")
    feature_importance, roc_points, calibration_points = _build_diagnostics(
        calibrated_model, X_test_scaled, y_test, y_prob
    )

    # --- What the shipped threshold does, and to whom ---
    operating_point = operating_point_metrics(y_test, y_prob)
    subgroups = subgroup_metrics(X_test, y_test, y_prob)
    cv_report = repeated_cross_validation(X_train, y_train)

    print("\n" + "=" * 50)
    print("CARDIOVASCULAR RISK MODEL PERFORMANCE")
    print("=" * 50)
    print(f"Accuracy:    {accuracy:.2f}%")
    print(f"ROC-AUC:     {roc_auc:.2f}%")
    print(f"Brier Score: {brier:.4f} (lower is better, calibrated)")
    print(f"Baseline LR: acc {baseline_metrics['accuracy']}% | auc {baseline_metrics['roc_auc']}% | brier {baseline_metrics['brier_score']}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    print(f"Cross-validated (95% CI over {cv_report['folds']} folds):")
    for metric in ("accuracy", "roc_auc", "brier_score"):
        stats = cv_report[metric]
        print(
            f"  {metric:<12} {stats['mean']:.4g} "
            f"[{stats['ci_low']:.4g}, {stats['ci_high']:.4g}]"
        )

    print(f"\nAt the shipped threshold of {operating_point['threshold']:.2f} "
          f"(prevalence {operating_point['prevalence']}%):")
    print(f"  sensitivity {operating_point['sensitivity']}% | specificity {operating_point['specificity']}%")
    print(f"  PPV         {operating_point['ppv']}% | NPV         {operating_point['npv']}%")

    print("\nSubgroups:")
    for group in subgroups:
        print(
            f"  {group['dimension']:<9} {group['group']:<13} n={group['n']:<6} "
            f"auc {group['roc_auc']} | sens {group['sensitivity']} | spec {group['specificity']}"
        )

    bundle = {
        "model": calibrated_model,
        "scaler": scaler,
        "baseline_model": baseline_model,
        "feature_names": FEATURE_NAMES,
        "feature_labels": FEATURE_LABELS,
        "metrics": {
            "accuracy": round(accuracy, 2),
            "roc_auc": round(roc_auc, 2),
            "brier_score": round(brier, 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        },
        "cross_validation": cv_report,
        "operating_point": operating_point,
        "subgroup_metrics": subgroups,
        "feature_importance": feature_importance,
        "roc_curve": roc_points,
        "calibration_curve": calibration_points,
        "baseline_metrics": baseline_metrics,
        "dataset": {
            "name": "UCI Cardiovascular Disease (70,000 patients)",
            "rows_used": len(df),
            "source": "https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset",
        },
    }

    joblib.dump(bundle, MODEL_SAVE_PATH)
    print(f"\nModel successfully saved to: {MODEL_SAVE_PATH}")
    return bundle


_BUNDLE_CACHE: Dict[str, Any] = {}


def load_model_bundle(force_reload: bool = False) -> Dict[str, Any]:
    """Loads the trained artifact from disk (cached in-process).

    Inference NEVER triggers training. If the artifact is missing the caller gets
    an actionable error instead of a multi-minute blocking request.
    """
    if not force_reload and "bundle" in _BUNDLE_CACHE:
        return _BUNDLE_CACHE["bundle"]

    if not MODEL_SAVE_PATH.exists():
        raise ModelNotTrainedError(
            f"No trained model found at {MODEL_SAVE_PATH}. "
            "Run 'python -m src.train' once to train and persist the model before starting the server."
        )

    bundle = joblib.load(MODEL_SAVE_PATH)
    _BUNDLE_CACHE["bundle"] = bundle
    return bundle


def model_info() -> Dict[str, Any]:
    """Returns honest model metadata: metrics, baseline comparison and diagnostics."""
    bundle = load_model_bundle()
    return {
        "algorithm": "HistGradientBoostingClassifier + sigmoid probability calibration (5-fold CV)",
        "feature_names": bundle["feature_names"],
        "feature_labels": bundle["feature_labels"],
        "metrics": bundle["metrics"],
        "cross_validation": bundle.get("cross_validation"),
        "operating_point": bundle.get("operating_point"),
        "subgroup_metrics": bundle.get("subgroup_metrics", []),
        "baseline_metrics": bundle["baseline_metrics"],
        "feature_importance": bundle["feature_importance"],
        "roc_curve": bundle["roc_curve"],
        "calibration_curve": bundle["calibration_curve"],
        "dataset": bundle["dataset"],
        "caveats": [
            "Trained on a single public dataset; not validated on local patient populations.",
            "Symptom interpretation is rule-based (auditable regex), not a language model.",
            "Output is a screening aid only and is not a medical diagnosis.",
        ],
    }


def predict_patient_risk(vitals: dict) -> dict:
    """Predicts calibrated cardiovascular risk for given patient vitals.

    Accepts:
    - age (years)
    - gender (1=female, 2=male)
    - height (cm)
    - weight (kg)
    - ap_hi (systolic BP, mm Hg)
    - ap_lo (diastolic BP, mm Hg)
    - cholesterol (1=normal, 2=above, 3=high) -- or None if unknown
    - gluc (1=normal, 2=above, 3=high) -- or None if unknown
    - smoke (0 or 1)
    - alco (0 or 1) -- or None if unknown
    - active (0 or 1) -- or None if unknown
    """
    bundle = load_model_bundle()
    model = bundle["model"]
    scaler = bundle["scaler"]

    # Calculate derived attributes
    age_years = float(vitals["age"])
    gender = int(vitals.get("gender", 1))
    height = float(vitals.get("height", 165))
    weight = float(vitals.get("weight", 70))
    bmi = weight / ((height / 100.0) ** 2)

    ap_hi = float(vitals.get("ap_hi", 120))
    ap_lo = float(vitals.get("ap_lo", 80))
    pulse_pressure = ap_hi - ap_lo

    # A self-reported value the patient does not know stays missing (NaN).
    # HistGradientBoosting handles missing values natively, so the model falls
    # back on what the patient *did* report instead of being handed an invented
    # "normal" reading, which used to bias every uncertain patient downwards.
    def _reported(key, default=None):
        value = vitals.get(key, default)
        return np.nan if value is None else float(value)

    chol = _reported("cholesterol")
    gluc = _reported("gluc")
    smoke = _reported("smoke", 0)
    alco = _reported("alco")
    active = _reported("active")

    # Named columns, not a bare array: the scaler was fitted on a DataFrame and
    # sklearn warns (and will eventually refuse) unnamed input.
    feature_row = pd.DataFrame(
        [[age_years, gender, bmi, ap_hi, ap_lo, chol, gluc, smoke, alco, active]],
        columns=FEATURE_NAMES,
    )

    scaled_row = scaler.transform(feature_row)
    pred_class = int(model.predict(scaled_row)[0])
    prob_percent = float(model.predict_proba(scaled_row)[0][1] * 100)

    # Determine risk category & theme
    if prob_percent < 25.0:
        risk_category = "Optimal / Low Risk (Safe)"
        color = "#10b981"  # Emerald green
        theme = "green"
    elif prob_percent < 60.0:
        risk_category = "Moderate Risk"
        color = "#eab308"  # Yellow / Amber
        theme = "yellow"
    elif prob_percent < 80.0:
        risk_category = "In Risk (Elevated)"
        color = "#ef4444"  # Red
        theme = "red"
    else:
        risk_category = "High Extreme Risk"
        color = "#7f1d1d"  # Blood Red
        theme = "blood-red"

    return {
        "probability": round(prob_percent, 1),
        "predicted_class": pred_class,
        "risk_category": risk_category,
        "color": color,
        "theme": theme,
        "bmi": round(bmi, 1),
        "pulse_pressure": round(pulse_pressure, 1),
    }


if __name__ == "__main__":
    train_model(force=True)
