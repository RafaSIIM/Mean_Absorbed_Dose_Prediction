"""
train_lasso_infold_models.py
=============================

LASSO-in-fold feature selection + Bayesian hyperparameter optimisation +
Leave-One-Patient-Out cross-validation (LOPOCV) + Jackknife+ conformal
prediction, for VOI-level absorbed-dose regression in [177Lu]Lu-PSMA-617
radioligand therapy.

Methodological note (Methodology 2, fold-independent feature selection):
--------------------------------------------------------------------------
Unlike a pipeline that consumes a fixed, pre-selected feature subset (LASSO
or BORUTA applied once on the full dataset before cross-validation), this
script repeats LASSO feature selection independently *inside every LOPOCV
training fold*, using only that fold's training data. This is the more
conservative of the two feature-selection protocols compared in the
companion dissertation (see Section 4.4.4, "Methodology 1" vs.
"Methodology 2"): it prevents the feature selector from ever seeing the
held-out patient, at the cost of a smaller and more variable feature subset
per fold and typically lower (but less optimistically biased) performance
estimates than pre-selection.

For each dose-integration method (Exponential / Trapezoidal) and each
configured algorithm, the script reports:

    - R^2, Pearson's r, MAE, RMSE, MAPE
    - Lin's Concordance Correlation Coefficient (CCC)
    - Intraclass Correlation Coefficient, ICC(2,1), absolute agreement
    - Bland-Altman bias and limits of agreement (LoA)
    - Number of LASSO-selected features per fold (mean +/- SD)
    - Jackknife+ conformal prediction intervals

Outputs (per dose-integration method, per algorithm): predicted-vs-actual
plots, Bland-Altman plots, a SHAP summary plot (fitted on the full dataset,
for visualisation only), and CSV tables with per-patient predictions and
aggregate performance metrics.

--------------------------------------------------------------------------
DATA & ETHICS DISCLAIMER
--------------------------------------------------------------------------
This repository does NOT include patient-level data. The input CSV
referenced in INPUT_FILE below contains de-identified radiomic features and
clinical biomarkers collected under an institutional ethics approval and is
not publicly distributable. To reuse this script, point INPUT_FILE to your
own fused PET+CT (+ clinical biomarker) feature file, structured with the
same column layout (see "Expected input schema" below).

Expected input schema (INPUT_FILE):
    'Patient'                                                    -- patient ID (str)
    'ROI'                                                        -- region-of-interest label (optional)
    'Absorbed Dose per Injected Activity (Gy/GBq) Exponential'  -- target (Exponential fit)
    'Absorbed Dose per Injected Activity (Gy/GBq) Trapezoid'    -- target (Trapezoidal fit)
    <feature_1>, <feature_2>, ...                                -- full (unselected) radiomic
                                                                     and/or clinical biomarker features

Unlike `train_ml_models_post_selection.py`, INPUT_FILE here should be the
*full* fused feature set (before any LASSO/BORUTA reduction), since feature
selection is performed internally, per fold, by this script.

Author: Rafael Tiago Morais Simões
Thesis: Pre-treatment prediction of absorbed dose distributions in patients
        undergoing personalized [177Lu]Lu-PSMA therapy, NOVA University Lisbon.
"""

import os
import warnings
from typing import Optional, Union

os.environ["PYTHONHASHSEED"] = "42"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import optuna
import shap
from scipy.stats import pearsonr

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.linear_model import LassoCV, BayesianRidge, LinearRegression, HuberRegressor, ElasticNet
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)


# ══════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION
# Edit the parameters below to run this script on a different VOI, dataset,
# normalisation scheme, or algorithm subset. No other changes should be
# required to reproduce results for a new organ or lesion cohort.
# ══════════════════════════════════════════════════════════════════════════

#: Volume of interest being modelled. Used only for labelling plots, output
#: folders, and file names — does not affect the modelling logic itself.
#: Typical values: "Kidneys", "Liver", "Spleen", "Lesions".
VOI_NAME: str = "Liver"

#: Path to the FULL fused PET+CT (+ clinical biomarker) feature dataset for
#: this VOI — i.e. before any LASSO/BORUTA reduction, since LASSO feature
#: selection is performed internally by this script, independently within
#: every cross-validation fold.
INPUT_FILE: str = "Dataset_ML_Liver_FUSED.csv"

#: Dose-integration methods to evaluate. Both are present as separate
#: target columns in INPUT_FILE (see "Expected input schema" above).
DOSE_METHODS: list[str] = ["Exponential", "Trapezoid"]

#: Feature-scaling strategy applied by the in-fold LASSO selector (fit on
#: the training fold only, to avoid leakage into the held-out fold).
#: One of: "standard" | "robust" | "power" | "none".
NORMALIZATION_METHOD: str = "standard"

#: Regression algorithms to benchmark, in the order they should be run.
#: Must be a subset of the keys defined in `ALGORITHMS` below:
#:   "Bayesian_Ridge", "Linear_Regression", "Huber_Regressor", "ElasticNet".
ALGORITHMS_TO_RUN: list[str] = [
    "Bayesian_Ridge",
    "Linear_Regression",
    "Huber_Regressor",
    "ElasticNet",
]

#: Number of Optuna trials per LASSO-in-fold hyperparameter search
#: (ignored for "Linear_Regression", which has no tunable hyperparameters).
N_TRIALS: int = 30

#: Miscoverage rate for the Jackknife+ prediction interval
#: (0.10 -> nominal 90% coverage).
ALPHA_CI: float = 0.10

#: Global random seed, applied to NumPy, Optuna, and every stochastic
#: scikit-learn estimator for reproducibility.
RANDOM_SEED: int = 42

#: Root directory for all generated plots and CSV outputs.
OUTPUT_ROOT: str = f"Plots_{VOI_NAME}_LASSO_IN_FOLD"

np.random.seed(RANDOM_SEED)
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ══════════════════════════════════════════════════════════════════════════
# SCALING
# ══════════════════════════════════════════════════════════════════════════

def get_scaler(method: str) -> Union[StandardScaler, RobustScaler, PowerTransformer]:
    """Instantiate the requested feature-scaling transformer.

    Args:
        method: One of "standard", "robust", "power", "none" (case-insensitive).
            - "standard": zero mean, unit variance (StandardScaler).
            - "robust": median/IQR-based scaling, less sensitive to outliers
              (RobustScaler).
            - "power": Yeo-Johnson power transform towards a Gaussian-like
              distribution (PowerTransformer).
            - "none": returns a no-op StandardScaler (`with_mean=False,
              with_std=False`), compatible with the manual `.transform()`
              calls used throughout this script (the LASSO-in-fold pipeline
              here is built by hand rather than via `sklearn.Pipeline`).

    Returns:
        A fresh, unfitted scikit-learn transformer instance.

    Raises:
        ValueError: If `method` is not a recognised option.
    """
    method_key = method.strip().lower()
    if method_key == "standard":
        return StandardScaler()
    if method_key == "robust":
        return RobustScaler()
    if method_key == "power":
        return PowerTransformer(method="yeo-johnson")
    if method_key == "none":
        return StandardScaler(with_mean=False, with_std=False)
    raise ValueError(
        f"Unknown NORMALIZATION_METHOD '{method}'. "
        "Choose one of: 'standard', 'robust', 'power', 'none'."
    )


# ══════════════════════════════════════════════════════════════════════════
# WINDOWS LONG-PATH HELPERS
# On Windows, file paths longer than 260 characters raise OSError unless
# explicitly prefixed. These helpers are a no-op on other platforms.
# ══════════════════════════════════════════════════════════════════════════

def _win_long_path(path: str) -> str:
    """Prefix a path with `\\\\?\\` to bypass the Windows 260-character limit.

    Args:
        path: An absolute file or directory path.

    Returns:
        The same path, prefixed for Windows long-path support if running on
        Windows and not already prefixed; unchanged on other platforms.
    """
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + path
    return path


def ensure_dir(path: str) -> str:
    """Create a directory (using an absolute, long-path-safe path) if needed.

    Args:
        path: A relative or absolute directory path.

    Returns:
        The absolute path to the (now-existing) directory.
    """
    abs_path = os.path.abspath(path)
    os.makedirs(_win_long_path(abs_path), exist_ok=True)
    return abs_path


def safe_savefig(fig: plt.Figure, output_dir: str, filename: str, dpi: int = 300, **kwargs) -> None:
    """Save a matplotlib figure using a long-path-safe absolute path.

    Args:
        fig: The figure to save.
        output_dir: Target directory (created if missing).
        filename: Output file name, including extension.
        dpi: Resolution in dots per inch.
        **kwargs: Additional keyword arguments forwarded to `fig.savefig`.
    """
    abs_dir = ensure_dir(output_dir)
    full_path = _win_long_path(os.path.join(abs_dir, filename))
    fig.savefig(full_path, dpi=dpi, **kwargs)
    plt.close(fig)


def safe_to_csv(df: pd.DataFrame, output_dir: str, filename: str) -> None:
    """Save a DataFrame to CSV using a long-path-safe absolute path.

    Args:
        df: The DataFrame to save.
        output_dir: Target directory (created if missing).
        filename: Output file name, including extension.
    """
    abs_dir = ensure_dir(output_dir)
    full_path = _win_long_path(os.path.join(abs_dir, filename))
    df.to_csv(full_path, index=False)


# ══════════════════════════════════════════════════════════════════════════
# AGREEMENT / CONCORDANCE METRICS
# ══════════════════════════════════════════════════════════════════════════

def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Lin's Concordance Correlation Coefficient (CCC).

    Args:
        y_true: Ground-truth (measured) absorbed doses.
        y_pred: Model-predicted absorbed doses.

    Returns:
        The CCC value (range -1 to 1, with 1 indicating perfect agreement),
        or 0.0 if the denominator is numerically degenerate.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mean_t, mean_p = np.mean(y_true), np.mean(y_pred)
    var_t, var_p = np.var(y_true), np.var(y_pred)
    covar = np.mean((y_true - mean_t) * (y_pred - mean_p))
    denom = var_t + var_p + (mean_t - mean_p) ** 2
    return float(2 * covar / denom) if denom > 1e-12 else 0.0


def intraclass_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute ICC(2,1): two-way mixed-effects, absolute agreement, single rater.

    Implements the McGraw & Wong (1996) formulation directly, with no
    external dependency, treating the measured and predicted values as two
    "raters" of the same underlying quantity.

    Args:
        y_true: Ground-truth (measured) absorbed doses.
        y_pred: Model-predicted absorbed doses.

    Returns:
        The ICC(2,1) value, or 0.0 if the denominator is numerically degenerate.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    data = np.column_stack([y_true, y_pred])
    grand_mean = data.mean()

    ss_total = np.sum((data - grand_mean) ** 2)
    ss_rows = 2 * np.sum((data.mean(axis=1) - grand_mean) ** 2)
    ss_cols = n * np.sum((data.mean(axis=0) - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / (n - 1)
    ms_cols = ss_cols

    denom = ms_rows + ms_error + (2 / n) * (ms_cols - ms_error)
    return float((ms_rows - ms_error) / denom) if abs(denom) > 1e-12 else 0.0


def bland_altman_stats(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    """Compute Bland-Altman agreement statistics.

    Args:
        y_true: Ground-truth (measured) absorbed doses.
        y_pred: Model-predicted absorbed doses.

    Returns:
        A tuple (bias, loa_lower, loa_upper, pct_within_loa).
    """
    diff = np.asarray(y_pred) - np.asarray(y_true)
    bias = diff.mean()
    sd = diff.std()
    loa_lower = bias - 1.96 * sd
    loa_upper = bias + 1.96 * sd
    pct = np.mean((diff >= loa_lower) & (diff <= loa_upper)) * 100
    return bias, loa_lower, loa_upper, pct


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAPE while excluding near-zero ground-truth values.

    Standard MAPE is undefined (division by zero) when `y_true` contains
    values at or near zero, which occurs for a subset of low-dose kidney and
    liver measurements in this cohort. Such samples are excluded from the
    MAPE calculation only (they remain included in every other metric).

    Args:
        y_true: Ground-truth (measured) absorbed doses.
        y_pred: Model-predicted absorbed doses.

    Returns:
        MAPE computed over the subset of samples with |y_true| > 1e-8, or
        NaN if no such samples exist.
    """
    y_true = np.asarray(y_true, dtype=float)
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return float("nan")
    return mean_absolute_percentage_error(y_true[mask], np.asarray(y_pred)[mask])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute the full set of regression and agreement metrics.

    Args:
        y_true: Ground-truth (measured) absorbed doses.
        y_pred: Model-predicted absorbed doses.

    Returns:
        A dictionary with keys: r2, mae, rmse, mape, pearson_r, ccc, icc,
        ba_bias, ba_lo, ba_hi, ba_pct.
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    ccc = concordance_correlation_coefficient(y_true, y_pred)
    icc = intraclass_correlation_coefficient(y_true, y_pred)
    ba_bias, ba_lo, ba_hi, ba_pct = bland_altman_stats(y_true, y_pred)
    return dict(
        r2=r2, mae=mae, rmse=rmse, mape=mape, pearson_r=pearson_r,
        ccc=ccc, icc=icc, ba_bias=ba_bias, ba_lo=ba_lo, ba_hi=ba_hi, ba_pct=ba_pct,
    )


# ══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════

def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    method_name: str,
    output_dir: str,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
) -> None:
    """Save a predicted-vs-actual scatter plot with the identity line.

    If `y_lower`/`y_upper` are provided (Jackknife+ prediction interval
    bounds), points are drawn with asymmetric error bars instead of plain
    markers.

    Args:
        y_true: Ground-truth (measured) absorbed doses.
        y_pred: Model-predicted absorbed doses.
        model_name: Algorithm name, used in the title and output filename.
        method_name: Dose-integration method, used in the title and filename.
        output_dir: Directory where the PNG file will be saved.
        y_lower: Optional lower bound of the prediction interval per sample.
        y_upper: Optional upper bound of the prediction interval per sample.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    if y_lower is not None and y_upper is not None:
        ax.errorbar(
            y_true, y_pred,
            yerr=[y_pred - y_lower, y_upper - y_pred],
            fmt="o", alpha=0.6, ecolor="steelblue", capsize=3,
            markersize=4, markeredgecolor="k", markeredgewidth=0.4,
            label="Prediction interval (90%)",
        )
    else:
        ax.scatter(y_true, y_pred, alpha=0.7, s=40, edgecolors="k", linewidths=0.4)

    lo = min(np.min(y_true), np.min(y_pred)) * 0.9
    hi = max(np.max(y_true), np.max(y_pred)) * 1.1
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Identity (y=x)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    ccc = concordance_correlation_coefficient(y_true, y_pred)

    ax.set_xlabel("Actual Dose (Gy/GBq)", fontsize=10)
    ax.set_ylabel("Predicted Dose (Gy/GBq)", fontsize=10)
    ax.set_title(
        f"{model_name} [{NORMALIZATION_METHOD}]\n{method_name} | R\u00b2={r2:.3f} | "
        f"MAE={mae:.4f} | r={pearson_r:.3f} | CCC={ccc:.3f}",
        fontsize=9,
    )
    ax.legend(fontsize=7)
    plt.tight_layout()

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)[:40]
    safe_savefig(fig, output_dir, f"PvA_{method_name}_{safe_name}.png")


def plot_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    method_name: str,
    output_dir: str,
) -> None:
    """Save a Bland-Altman agreement plot.

    Args:
        y_true: Ground-truth (measured) absorbed doses.
        y_pred: Model-predicted absorbed doses.
        model_name: Algorithm name, used in the title and output filename.
        method_name: Dose-integration method, used in the title and filename.
        output_dir: Directory where the PNG file will be saved.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mean_val = (y_true + y_pred) / 2
    diff = y_pred - y_true
    bias = diff.mean()
    loa = 1.96 * diff.std()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(mean_val, diff, alpha=0.7, s=40, edgecolors="k", linewidths=0.4)
    ax.axhline(bias, color="red", linestyle="--", linewidth=1, label=f"Bias={bias:.4f}")
    ax.axhline(bias + loa, color="blue", linestyle=":", linewidth=1, label=f"+1.96SD={bias + loa:.4f}")
    ax.axhline(bias - loa, color="blue", linestyle=":", linewidth=1, label=f"-1.96SD={bias - loa:.4f}")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Mean Actual & Predicted (Gy/GBq)", fontsize=10)
    ax.set_ylabel("Predicted - Actual (Gy/GBq)", fontsize=10)
    ax.set_title(f"Bland-Altman: {model_name} [{NORMALIZATION_METHOD}]\n{method_name}", fontsize=9)
    ax.legend(fontsize=7)
    plt.tight_layout()

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)[:40]
    safe_savefig(fig, output_dir, f"BA_{method_name}_{safe_name}.png")


def plot_shap(model, model_name: str, X_df: pd.DataFrame, method_name: str, output_dir: str) -> None:
    """Save a SHAP summary plot using a sampled KernelExplainer.

    A model-agnostic KernelExplainer is used unconditionally here (rather
    than dispatching by model type) because all four supported algorithms
    (Bayesian Ridge, Linear Regression, Huber Regressor, ElasticNet) are
    linear estimators for which KernelExplainer remains tractable at this
    sample size; this also keeps the explainer logic identical regardless
    of which algorithm produced the LASSO-in-fold-selected feature subset.

    Args:
        model: A fitted regressor.
        model_name: Algorithm name, used in the title and output filename.
        X_df: Feature matrix (already imputed/scaled/LASSO-selected) as a
            DataFrame, with column names matching the fitted model's inputs.
        method_name: Dose-integration method, used in the title and filename.
        output_dir: Directory where the PNG file will be saved.
    """
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)[:40]
    try:
        background = shap.sample(X_df, min(50, len(X_df)), random_state=RANDOM_SEED)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_df, nsamples=100)

        fig, _ = plt.subplots(figsize=(8, max(4, 0.4 * X_df.shape[1])))
        shap.summary_plot(shap_values, X_df, show=False, plot_size=None)
        plt.title(
            f"SHAP [{NORMALIZATION_METHOD}]: {model_name} | {method_name}\n"
            f"(visualisation only — fitted on full dataset)",
            fontsize=9,
        )
        plt.tight_layout()
        safe_savefig(fig, output_dir, f"SHAP_{method_name}_{safe_name}.png", dpi=300, bbox_inches="tight")
    except Exception as exc:  # noqa: BLE001 -- SHAP's KernelExplainer can fail for many numerical reasons
        plt.close("all")
        print(f"      SHAP failed for {model_name}: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# IN-FOLD LASSO FEATURE SELECTOR
# ══════════════════════════════════════════════════════════════════════════

def fit_lasso_selector(X_train: np.ndarray, y_train: np.ndarray):
    """Fit an imputer, scaler, and LASSO-based feature selection mask on training data only.

    This is the core "in-fold" step: called separately inside every LOPOCV
    training fold (and once more on the full dataset for the final SHAP
    visualisation), so the feature subset can legitimately differ across
    folds and no information from the held-out patient ever influences
    which features are selected.

    Args:
        X_train: Training-fold feature matrix (raw, unscaled).
        y_train: Training-fold target vector.

    Returns:
        A tuple (mask, imputer, scaler):
            - mask: Boolean array selecting features with a non-zero
              LassoCV coefficient (falls back to the top-5 features by
              |coefficient| if LASSO shrinks all coefficients to zero).
            - imputer: Fitted `SimpleImputer` (median strategy).
            - scaler: Fitted scaler, per `NORMALIZATION_METHOD`.
    """
    imputer = SimpleImputer(strategy="median")
    scaler = get_scaler(NORMALIZATION_METHOD)
    X_imp = imputer.fit_transform(X_train)
    X_scaled = scaler.fit_transform(X_imp)

    lasso = LassoCV(cv=5, max_iter=20000, random_state=RANDOM_SEED, n_jobs=-1)
    lasso.fit(X_scaled, y_train)

    mask = np.abs(lasso.coef_) > 1e-6
    if mask.sum() == 0:
        top5 = np.argsort(np.abs(lasso.coef_))[-5:]
        mask[top5] = True

    return mask, imputer, scaler


# ══════════════════════════════════════════════════════════════════════════
# OPTUNA OBJECTIVES & ALGORITHM REGISTRY
# ══════════════════════════════════════════════════════════════════════════

def _cv_mae(model, X_tr: np.ndarray, y_tr: np.ndarray) -> float:
    """Compute mean 3-fold inner-CV MAE for a candidate model instance.

    Args:
        model: An unfitted scikit-learn regressor.
        X_tr: Training-fold, LASSO-selected, scaled feature matrix.
        y_tr: Training-fold target vector.

    Returns:
        Mean absolute error averaged across the 3 inner folds, or
        `float("inf")` if fitting fails for any inner fold.
    """
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    maes = []
    for tr_idx, val_idx in inner_cv.split(X_tr):
        m = clone(model)
        try:
            m.fit(X_tr[tr_idx], y_tr[tr_idx])
            maes.append(mean_absolute_error(y_tr[val_idx], m.predict(X_tr[val_idx])))
        except Exception:
            return float("inf")
    return float(np.mean(maes))


def objective_bayesian_ridge(trial: optuna.Trial, X_tr: np.ndarray, y_tr: np.ndarray) -> float:
    """Optuna objective for Bayesian Ridge Regression.

    Args:
        trial: The active Optuna trial.
        X_tr: Training-fold, LASSO-selected, scaled feature matrix.
        y_tr: Training-fold target vector.

    Returns:
        Mean inner-CV MAE for the sampled hyperparameters.
    """
    model = BayesianRidge(
        alpha_1=trial.suggest_float("alpha_1", 1e-6, 1e-2, log=True),
        alpha_2=trial.suggest_float("alpha_2", 1e-6, 1e-2, log=True),
        lambda_1=trial.suggest_float("lambda_1", 1e-6, 1e-2, log=True),
        lambda_2=trial.suggest_float("lambda_2", 1e-6, 1e-2, log=True),
    )
    return _cv_mae(model, X_tr, y_tr)


def objective_linear_regression(trial: optuna.Trial, X_tr: np.ndarray, y_tr: np.ndarray) -> float:
    """Optuna objective for Ordinary Least Squares (no tunable hyperparameters).

    Args:
        trial: The active Optuna trial (unused; kept for interface consistency).
        X_tr: Training-fold, LASSO-selected, scaled feature matrix.
        y_tr: Training-fold target vector.

    Returns:
        Mean inner-CV MAE (identical across trials, since there is nothing to tune).
    """
    return _cv_mae(LinearRegression(), X_tr, y_tr)


def objective_huber(trial: optuna.Trial, X_tr: np.ndarray, y_tr: np.ndarray) -> float:
    """Optuna objective for the Huber Regressor.

    Args:
        trial: The active Optuna trial.
        X_tr: Training-fold, LASSO-selected, scaled feature matrix.
        y_tr: Training-fold target vector.

    Returns:
        Mean inner-CV MAE for the sampled hyperparameters.
    """
    model = HuberRegressor(
        epsilon=trial.suggest_float("epsilon", 1.35, 5.0),
        alpha=trial.suggest_float("alpha", 1e-5, 1.0, log=True),
        max_iter=5000,
        tol=1e-2,
    )
    return _cv_mae(model, X_tr, y_tr)


def objective_elasticnet(trial: optuna.Trial, X_tr: np.ndarray, y_tr: np.ndarray) -> float:
    """Optuna objective for ElasticNet Regression.

    Args:
        trial: The active Optuna trial.
        X_tr: Training-fold, LASSO-selected, scaled feature matrix.
        y_tr: Training-fold target vector.

    Returns:
        Mean inner-CV MAE for the sampled hyperparameters.
    """
    model = ElasticNet(
        alpha=trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        l1_ratio=trial.suggest_float("l1_ratio", 0.01, 0.99),
        max_iter=20000,
        random_state=RANDOM_SEED,
    )
    return _cv_mae(model, X_tr, y_tr)


def _huber_fallback(X_tr: np.ndarray, y_tr: np.ndarray):
    """Progressively relax convergence tolerance for a diverging Huber Regressor.

    HuberRegressor occasionally fails to converge on small, high-dimensional
    folds. This helper retries with increasingly permissive `tol`/`max_iter`
    settings before falling back to Ordinary Least Squares as a last resort.

    Args:
        X_tr: Training-fold, LASSO-selected, scaled feature matrix.
        y_tr: Training-fold target vector.

    Returns:
        A fitted `HuberRegressor` or, failing that, a fitted `LinearRegression`.
    """
    for tol, max_iter in [(1e-1, 5000), (5e-1, 10000), (1.0, 20000)]:
        try:
            m = HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=max_iter, tol=tol)
            m.fit(X_tr, y_tr)
            return m
        except Exception:
            continue
    print("      Warning: Huber Regressor fallback exhausted -> using Linear Regression")
    m = LinearRegression()
    m.fit(X_tr, y_tr)
    return m


def _fit_with_fallback(algo_name: str, build_fn, best_params: dict, X_tr: np.ndarray, y_tr: np.ndarray):
    """Fit the requested model, falling back to Huber-specific or OLS recovery on failure.

    Centralises the "try the tuned model, else fall back" logic that is
    otherwise duplicated across the LOPOCV loop, the Jackknife+ loop, and
    the final SHAP fit, avoiding the risk of the two diverging (as happened
    in an earlier draft of this script).

    Args:
        algo_name: One of the keys of `ALGORITHMS`.
        build_fn: The `"build_model"` callable from `ALGORITHMS[algo_name]`.
        best_params: Hyperparameters to pass to `build_fn`.
        X_tr: Training-fold, LASSO-selected, scaled feature matrix.
        y_tr: Training-fold target vector.

    Returns:
        A fitted regressor: either the requested model, a tolerance-relaxed
        Huber Regressor, or Ordinary Least Squares, in that order of preference.
    """
    try:
        model = build_fn(best_params)
        model.fit(X_tr, y_tr)
        return model
    except Exception:
        if algo_name == "Huber_Regressor":
            return _huber_fallback(X_tr, y_tr)
        model = LinearRegression()
        model.fit(X_tr, y_tr)
        return model


#: Registry mapping each supported algorithm name to its Optuna objective,
#: model-building function, and number of hyperparameter search trials.
ALGORITHMS: dict = {
    "Bayesian_Ridge": {
        "objective": objective_bayesian_ridge,
        "build_model": lambda p: BayesianRidge(**p),
        "n_trials": N_TRIALS,
    },
    "Linear_Regression": {
        "objective": objective_linear_regression,
        "build_model": lambda p: LinearRegression(),
        "n_trials": 1,
    },
    "Huber_Regressor": {
        "objective": objective_huber,
        "build_model": lambda p: HuberRegressor(max_iter=5000, tol=1e-2, **p),
        "n_trials": N_TRIALS,
    },
    "ElasticNet": {
        "objective": objective_elasticnet,
        "build_model": lambda p: ElasticNet(max_iter=20000, random_state=RANDOM_SEED, **p),
        "n_trials": N_TRIALS,
    },
}


def validate_algorithms(algorithms: list[str]) -> None:
    """Raise early if `ALGORITHMS_TO_RUN` contains an unrecognised name.

    Args:
        algorithms: The user-configured list of algorithm names.

    Raises:
        ValueError: If any entry is not a key of `ALGORITHMS`.
    """
    unknown = [a for a in algorithms if a not in ALGORITHMS]
    if unknown:
        raise ValueError(
            f"Unsupported algorithm(s) in ALGORITHMS_TO_RUN: {unknown}. "
            f"Supported options are: {sorted(ALGORITHMS.keys())}"
        )


# ══════════════════════════════════════════════════════════════════════════
# JACKKNIFE+ CONFORMAL PREDICTION (WITH IN-FOLD LASSO)
# ══════════════════════════════════════════════════════════════════════════

def jackknife_plus(X: np.ndarray, y: np.ndarray, algo_name: str, alpha: float = ALPHA_CI):
    """Compute Jackknife+ leave-one-out predictions, repeating LASSO in-fold.

    For each held-out patient, LASSO feature selection, hyperparameter
    optimisation, and model fitting are all repeated from scratch on the
    remaining patients only, then used to predict the held-out point.

    Args:
        X: Full feature matrix (all patients, raw/unscaled).
        y: Full target vector (all patients).
        algo_name: One of the keys of `ALGORITHMS`.
        alpha: Miscoverage rate (e.g. 0.10 for a nominal 90% interval).

    Returns:
        A tuple (y_pred_loo, y_lower, y_upper): the leave-one-out
        predictions and their lower/upper conformal interval bounds.
    """
    algo = ALGORITHMS[algo_name]
    objective, build_fn, n_trials = algo["objective"], algo["build_model"], algo["n_trials"]

    X, y = np.asarray(X), np.asarray(y)
    n = len(y)
    loo = LeaveOneOut()
    y_pred_loo = np.zeros(n)
    residuals = np.zeros(n)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        mask, imputer, scaler = fit_lasso_selector(X_train, y_train)
        X_tr_sel = scaler.transform(imputer.transform(X_train))[:, mask]
        X_te_sel = scaler.transform(imputer.transform(X_test))[:, mask]

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        study.optimize(lambda trial: objective(trial, X_tr_sel, y_train), n_trials=n_trials, catch=(Exception,))

        model = _fit_with_fallback(algo_name, build_fn, study.best_params, X_tr_sel, y_train)

        i = test_idx[0]
        y_pred_loo[i] = model.predict(X_te_sel)[0]
        residuals[i] = abs(y[i] - y_pred_loo[i])

    q = np.quantile(residuals, 1 - alpha)
    return y_pred_loo, y_pred_loo - q, y_pred_loo + q


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_oar_data(file_path: str, method_name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load the full fused feature CSV and prepare it for in-fold LASSO selection.

    Args:
        file_path: Path to INPUT_FILE.
        method_name: Dose-integration method whose target column should be
            used ("Exponential" or "Trapezoid").

    Returns:
        A tuple (X_df, y, groups):
            - X_df: Feature DataFrame (sanitised column names, raw/unscaled values).
            - y: Target vector (absorbed dose, Gy/GBq).
            - groups: Patient ID per row, for group-aware cross-validation.
    """
    df = pd.read_csv(file_path)
    target_col = f"Absorbed Dose per Injected Activity (Gy/GBq) {method_name}"

    y_raw = pd.to_numeric(df[target_col].astype(str).str.replace(",", "."), errors="coerce")
    valid = y_raw.notna()
    df = df[valid].reset_index(drop=True)
    y = y_raw[valid].values
    groups = df["Patient"].values

    drop_cols = [
        "Patient", "ROI",
        "Absorbed Dose per Injected Activity (Gy/GBq) Exponential",
        "Absorbed Dose per Injected Activity (Gy/GBq) Trapezoid",
    ]
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    for col in X_df.columns:
        if X_df[col].dtype == "object":
            X_df[col] = pd.to_numeric(X_df[col].astype(str).str.replace(",", "."), errors="coerce")

    X_df.columns = (
        X_df.columns
        .str.replace("[", "_", regex=False).str.replace("]", "_", regex=False)
        .str.replace("<", "_", regex=False).str.replace(">", "_", regex=False)
        .str.replace("(", "_", regex=False).str.replace(")", "_", regex=False)
    )

    return X_df, y, groups


# ══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the full LASSO-in-fold benchmarking pipeline for the configured VOI.

    For every dose-integration method in `DOSE_METHODS` and every algorithm
    in `ALGORITHMS_TO_RUN`, runs LOPOCV with LASSO feature selection
    repeated inside every fold, exports per-algorithm plots/CSVs, and
    finally runs Jackknife+ (again repeating LASSO in-fold within the
    Jackknife+ leave-one-out loop) for conformal prediction intervals.
    """
    validate_algorithms(ALGORITHMS_TO_RUN)

    print("=" * 80)
    print(f" LASSO-IN-FOLD + {len(ALGORITHMS_TO_RUN)} ALGORITHM(S) + OPTUNA | VOI: {VOI_NAME}")
    print(f" Scaling: {NORMALIZATION_METHOD}")
    print(f" Algorithms: {', '.join(ALGORITHMS_TO_RUN)}")
    print(" Fitted on training-fold data only -- zero data leakage")
    print("=" * 80)

    all_results = []
    jackknife_results = []

    for method_name in DOSE_METHODS:
        print(f"\n{'=' * 80}")
        print(f" DOSE METHOD: {method_name.upper()}")
        print("=" * 80)

        X_df, y, groups = load_oar_data(INPUT_FILE, method_name)
        X = X_df.values
        feature_names = list(X_df.columns)
        n_patients = len(np.unique(groups))

        for algo_name in ALGORITHMS_TO_RUN:
            algo_cfg = ALGORITHMS[algo_name]
            objective, build_fn, n_trials = algo_cfg["objective"], algo_cfg["build_model"], algo_cfg["n_trials"]

            print(f"\n  -- ALGORITHM: {algo_name} --")

            out_base = os.path.abspath(os.path.join(OUTPUT_ROOT, method_name, algo_name))
            dirs = {k: ensure_dir(os.path.join(out_base, k)) for k in ("PredVsActual", "BlandAltman", "SHAP", "Predictions")}

            logo = LeaveOneGroupOut()
            y_true_all, y_pred_all, groups_all, n_features_per_fold = [], [], [], []

            print(f"    Running LOPOCV ({n_patients} patients)...")

            for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                mask, imputer, scaler = fit_lasso_selector(X_train, y_train)
                n_features_per_fold.append(int(mask.sum()))

                X_tr_sel = scaler.transform(imputer.transform(X_train))[:, mask]
                X_te_sel = scaler.transform(imputer.transform(X_test))[:, mask]

                study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
                study.optimize(lambda trial: objective(trial, X_tr_sel, y_train), n_trials=n_trials, catch=(Exception,))

                model_final = _fit_with_fallback(algo_name, build_fn, study.best_params, X_tr_sel, y_train)
                pred = model_final.predict(X_te_sel)

                y_true_all.extend(y_test.tolist())
                y_pred_all.extend(np.asarray(pred).tolist())
                groups_all.extend(groups[test_idx].tolist())

                pat = groups[test_idx][0]
                print(f"    Fold {fold_i + 1:02d} | Patient: {pat} | Features: {mask.sum()} | "
                      f"Pred: {pred[0]:.4f} | True: {y_test[0]:.4f}")

            y_true_arr, y_pred_arr, groups_arr = np.array(y_true_all), np.array(y_pred_all), np.array(groups_all)

            m = compute_metrics(y_true_arr, y_pred_arr)
            avg_feat, std_feat = float(np.mean(n_features_per_fold)), float(np.std(n_features_per_fold))

            print(f"\n    -- LOPOCV RESULTS ({method_name} | {algo_name}) [{NORMALIZATION_METHOD}]")
            print(f"       MAE:  {m['mae']:.4f} Gy/GBq | RMSE: {m['rmse']:.4f} Gy/GBq | MAPE: {m['mape']:.4f}")
            print(f"       R\u00b2: {m['r2']:+.3f} | Pearson r: {m['pearson_r']:.3f} | CCC: {m['ccc']:.3f} | ICC(2,1): {m['icc']:.3f}")
            print(f"       BA Bias: {m['ba_bias']:.4f} LoA: [{m['ba_lo']:.4f}, {m['ba_hi']:.4f}] ({m['ba_pct']:.1f}% within)")
            print(f"       Avg features/fold: {avg_feat:.1f} +/- {std_feat:.1f}")

            all_results.append({
                "VOI": VOI_NAME, "Scaling": NORMALIZATION_METHOD, "Dose_Method": method_name,
                "Algorithm": algo_name, "R2_Score": m["r2"], "Pearson_r": m["pearson_r"],
                "MAE_(Gy/GBq)": m["mae"], "RMSE_(Gy/GBq)": m["rmse"], "MAPE": m["mape"],
                "CCC": m["ccc"], "ICC(2,1)": m["icc"], "BA_Bias_(Gy/GBq)": m["ba_bias"],
                "BA_LoA_Lower": m["ba_lo"], "BA_LoA_Upper": m["ba_hi"], "BA_Pct_Within_LoA": m["ba_pct"],
                "Avg_Features_per_Fold": avg_feat,
            })

            plot_predicted_vs_actual(y_true_arr, y_pred_arr, algo_name, method_name, dirs["PredVsActual"])
            plot_bland_altman(y_true_arr, y_pred_arr, algo_name, method_name, dirs["BlandAltman"])

            safe_to_csv(
                pd.DataFrame({
                    "Patient": groups_arr, "y_true": y_true_arr, "y_pred": y_pred_arr,
                    "residual": y_pred_arr - y_true_arr,
                }),
                dirs["Predictions"], f"Predictions_{method_name}.csv",
            )

            print(f"\n    Running Jackknife+ ({method_name} | {algo_name})...")
            y_pred_jk, y_lower, y_upper = jackknife_plus(X, y, algo_name)

            coverage = float(np.mean((y >= y_lower) & (y <= y_upper)))
            interval_width = float(np.mean(y_upper - y_lower))
            mj = compute_metrics(y, y_pred_jk)

            print(f"       Coverage ({int((1 - ALPHA_CI) * 100)}%): {coverage:.1%} | "
                  f"Interval width: {interval_width:.4f} Gy/GBq")
            print(f"       MAE: {mj['mae']:.4f} | R\u00b2: {mj['r2']:.3f} | CCC: {mj['ccc']:.3f}")

            plot_predicted_vs_actual(y, y_pred_jk, f"{algo_name}_JK+", method_name, dirs["PredVsActual"], y_lower=y_lower, y_upper=y_upper)
            plot_bland_altman(y, y_pred_jk, f"{algo_name}_JK+", method_name, dirs["BlandAltman"])

            print("    Computing SHAP (fitted on full dataset -- visualisation only)...")
            mask_full, imp_full, scaler_full = fit_lasso_selector(X, y)
            X_full_sel = scaler_full.transform(imp_full.transform(X))[:, mask_full]
            feat_names_sel = [feature_names[i] for i in np.where(mask_full)[0]]

            shap_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
            shap_study.optimize(lambda trial: objective(trial, X_full_sel, y), n_trials=n_trials, catch=(Exception,))

            model_shap = _fit_with_fallback(algo_name, build_fn, shap_study.best_params, X_full_sel, y)
            plot_shap(model_shap, f"{algo_name}", pd.DataFrame(X_full_sel, columns=feat_names_sel), method_name, dirs["SHAP"])

            safe_to_csv(
                pd.DataFrame({
                    "Patient": groups, "y_true": y, "y_pred": y_pred_jk,
                    "lower_90": y_lower, "upper_90": y_upper,
                    "interval_width": y_upper - y_lower,
                    "within_interval": (y >= y_lower) & (y <= y_upper),
                }),
                dirs["Predictions"], f"JackknifePlus_{method_name}.csv",
            )

            jackknife_results.append({
                "VOI": VOI_NAME, "Scaling": NORMALIZATION_METHOD, "Dose_Method": method_name,
                "Algorithm": algo_name, "Coverage_90pct": coverage, "Mean_Interval_Width": interval_width,
                "MAE_(Gy/GBq)": mj["mae"], "R2_Score": mj["r2"], "Pearson_r": mj["pearson_r"],
                "CCC": mj["ccc"], "ICC(2,1)": mj["icc"], "BA_Bias_(Gy/GBq)": mj["ba_bias"],
                "BA_LoA_Lower": mj["ba_lo"], "BA_LoA_Upper": mj["ba_hi"], "BA_Pct_Within_LoA": mj["ba_pct"],
            })

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(f"ML_LASSOinFold_{NORMALIZATION_METHOD}_{VOI_NAME}_AllAlgos_Performance.csv", index=False)

    df_jk = pd.DataFrame(jackknife_results)
    df_jk.to_csv(f"ML_LASSOinFold_{NORMALIZATION_METHOD}_{VOI_NAME}_AllAlgos_JackknifePlus.csv", index=False)

    print(f"\n{'=' * 80}")
    print(f" GLOBAL RANKING -- {VOI_NAME} | {NORMALIZATION_METHOD} (sorted by MAE)")
    print("=" * 80)
    summary = df_results[["Dose_Method", "Algorithm", "MAE_(Gy/GBq)", "RMSE_(Gy/GBq)", "R2_Score", "CCC", "MAPE"]].copy()
    summary = summary.sort_values("MAE_(Gy/GBq)").reset_index(drop=True)
    print(summary.to_string(index=False))

    print(f"\n{'=' * 80}")
    print(f"PIPELINE COMPLETE -- VOI: {VOI_NAME} | {NORMALIZATION_METHOD} | {len(ALGORITHMS_TO_RUN)} algorithm(s)")
    print(f"Performance : ML_LASSOinFold_{NORMALIZATION_METHOD}_{VOI_NAME}_AllAlgos_Performance.csv")
    print(f"Jackknife+  : ML_LASSOinFold_{NORMALIZATION_METHOD}_{VOI_NAME}_AllAlgos_JackknifePlus.csv")
    print(f"Plots       : {OUTPUT_ROOT}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
