"""
train_ml_models_post_selection.py
==================================

Bayesian hyperparameter optimisation, Leave-One-Patient-Out cross-validation
(LOPOCV), and Jackknife+ conformal prediction intervals for VOI-level
absorbed-dose regression models in [177Lu]Lu-PSMA-617 radioligand therapy.

This script consumes the LASSO/BORUTA-selected feature datasets produced by
an upstream feature-selection step and benchmarks one or more regression
algorithms per dose-integration method (Exponential / Trapezoidal),
reporting:

    - R^2, Pearson's r, MAE, RMSE, MAPE
    - Lin's Concordance Correlation Coefficient (CCC)
    - Intraclass Correlation Coefficient, ICC(2,1), absolute agreement
    - Bland-Altman bias and limits of agreement (LoA)
    - Jackknife+ conformal prediction intervals for the best-performing model

Outputs (per dose-integration method): predicted-vs-actual plots (with a
shaded +/-20% error cone around the identity line), Bland-Altman plots,
SHAP summary plots, and CSV tables with per-patient predictions and
aggregate performance metrics.

--------------------------------------------------------------------------
DATA & ETHICS DISCLAIMER
--------------------------------------------------------------------------
This repository does NOT include patient-level data. The input CSV files
referenced in DATASET_PATHS below contain de-identified radiomic features
and clinical biomarkers collected under an institutional ethics approval and
are not publicly distributable. This code is shared to document the
modelling methodology; it is not intended to be run end-to-end without
adapting it to a dataset of your own (see "Expected input schema" below).

Expected input schema (per CSV):
    'Patient'                                                    -- patient ID (str)
    'ROI'                                                        -- region-of-interest label (optional)
    'Absorbed Dose per Injected Activity (Gy/GBq) Exponential'  -- target (Exponential fit)
    'Absorbed Dose per Injected Activity (Gy/GBq) Trapezoid'    -- target (Trapezoidal fit)
    <feature_1>, <feature_2>, ...                                -- LASSO/BORUTA-selected radiomic
                                                                     and/or clinical biomarker features

Author: Rafael Tiago Morais Simões
Thesis: Pre-treatment prediction of absorbed dose distributions in patients
        undergoing personalized [177Lu]Lu-PSMA therapy, NOVA University Lisbon.
"""

import os
import warnings
import textwrap
from typing import Optional, Union

os.environ["PYTHONHASHSEED"] = "42"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import optuna
import shap
from scipy.stats import pearsonr

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

from sklearn.linear_model import LinearRegression, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from xgboost import XGBRegressor


# ══════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION
# Edit the parameters below to run this script on a different VOI, dataset,
# normalisation scheme, or algorithm subset. No other changes should be
# required to reproduce results for a new organ or lesion cohort.
# ══════════════════════════════════════════════════════════════════════════

#: Volume of interest being modelled. Used only for labelling plots, output
#: folders, and file names — does not affect the modelling logic itself.
#: Typical values: "Kidneys", "Liver", "Spleen", "Lesions".
VOI_NAME: str = "Kidneys"

#: LASSO/BORUTA-selected feature datasets for this VOI, one per
#: dose-integration method, as produced by an upstream feature-selection
#: step. Edit these paths to point at your own exported CSVs.
DATASET_PATHS: dict[str, str] = {
    "Exponential": "Dataset_ML_Kidneys_FUSED_LASSO_Exponential.csv",
    "Trapezoid": "Dataset_ML_Kidneys_FUSED_LASSO_Trapezoid.csv",
}

#: Feature-scaling strategy applied inside every cross-validation fold
#: (fit on the training fold only, to avoid leakage into the held-out fold).
#: One of: "standard" | "robust" | "power" | "none".
NORMALIZATION_METHOD: str = "standard"

#: Regression algorithms to benchmark, in the order they should be run.
#: Supported names (see `build_model` / `optuna_objective` for definitions):
#:   "Linear Regression", "ElasticNet", "Bayesian Ridge", "Huber Regressor",
#:   "Random Forest", "Extra Trees", "Gradient Boosting", "XGBoost",
#:   "Support Vector Regressor (SVR)", "Decision Tree", "K-Neighbors (KNN)",
#:   "AdaBoost", "Gaussian Process"
ALGORITHMS: list[str] = [
    "Linear Regression",
    "ElasticNet",
    "Bayesian Ridge",
    "Huber Regressor",
]

#: Number of Optuna trials per inner-loop hyperparameter search.
N_TRIALS: int = 30

#: Miscoverage rate for the Jackknife+ prediction interval
#: (0.10 -> nominal 90% coverage).
JACKKNIFE_ALPHA: float = 0.10

#: Half-width of the shaded error cone drawn around the identity line on
#: every predicted-vs-actual plot (0.20 -> +/-20%).
ERROR_CONE_MARGIN: float = 0.20

#: Global random seed, applied to NumPy, Optuna, and every stochastic
#: scikit-learn / XGBoost estimator for reproducibility.
RANDOM_SEED: int = 42

#: Number of top features to display on each SHAP summary plot.
SHAP_MAX_DISPLAY: int = 10

#: Root directory for all generated plots and CSV outputs.
OUTPUT_ROOT: str = f"Plots_{VOI_NAME}_OPTUNA"

SUPPORTED_ALGORITHMS = {
    "Linear Regression", "ElasticNet", "Bayesian Ridge", "Huber Regressor",
    "Random Forest", "Extra Trees", "Gradient Boosting", "XGBoost",
    "Support Vector Regressor (SVR)", "Decision Tree", "K-Neighbors (KNN)",
    "AdaBoost", "Gaussian Process",
}

np.random.seed(RANDOM_SEED)
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ══════════════════════════════════════════════════════════════════════════
# SCALING
# ══════════════════════════════════════════════════════════════════════════

def get_scaler(method: str) -> Union[StandardScaler, RobustScaler, PowerTransformer, str]:
    """Instantiate the requested feature-scaling transformer.

    Args:
        method: One of "standard", "robust", "power", "none" (case-insensitive).
            - "standard": zero mean, unit variance (StandardScaler).
            - "robust": median/IQR-based scaling, less sensitive to outliers
              (RobustScaler).
            - "power": Yeo-Johnson power transform towards a Gaussian-like
              distribution (PowerTransformer).
            - "none": no scaling applied (returns the literal string
              "passthrough", understood natively by sklearn.Pipeline).

    Returns:
        A fresh scikit-learn transformer instance, or the string
        "passthrough" if no scaling should be applied.

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
        return "passthrough"
    raise ValueError(
        f"Unknown NORMALIZATION_METHOD '{method}'. "
        "Choose one of: 'standard', 'robust', 'power', 'none'."
    )


# ══════════════════════════════════════════════════════════════════════════
# AGREEMENT / CONCORDANCE METRICS
# ══════════════════════════════════════════════════════════════════════════

def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Lin's Concordance Correlation Coefficient (CCC).

    CCC jointly captures precision (Pearson correlation) and accuracy
    (deviation from the line of identity), ranging from -1 to 1, with 1
    indicating perfect agreement between predicted and measured values.

    Args:
        y_true: Ground-truth (measured) absorbed doses.
        y_pred: Model-predicted absorbed doses.

    Returns:
        The CCC value as a float.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    covar = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * covar) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return float(ccc)


def intraclass_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute ICC(2,1): two-way mixed-effects, absolute agreement, single rater.

    Implements the McGraw & Wong (1996) formulation directly, with no
    external dependency (e.g. pingouin), treating the measured and
    predicted values as two "raters" of the same underlying quantity.

    Args:
        y_true: Ground-truth (measured) absorbed doses.
        y_pred: Model-predicted absorbed doses.

    Returns:
        The ICC(2,1) value as a float.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    data = np.column_stack([y_true, y_pred])  # shape: (n_subjects, 2 raters)
    grand_mean = data.mean()

    ss_total = np.sum((data - grand_mean) ** 2)
    ss_rows = 2 * np.sum((data.mean(axis=1) - grand_mean) ** 2)   # between-subjects
    ss_cols = n * np.sum((data.mean(axis=0) - grand_mean) ** 2)   # between-raters
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / (n - 1)   # (n-1) * (k-1), with k=2
    ms_cols = ss_cols               # k-1 = 1

    icc = (ms_rows - ms_error) / (ms_rows + ms_error + (2 / n) * (ms_cols - ms_error))
    return float(icc)


def bland_altman_stats(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    """Compute Bland-Altman agreement statistics.

    Args:
        y_true: Ground-truth (measured) absorbed doses.
        y_pred: Model-predicted absorbed doses.

    Returns:
        A tuple (bias, loa_lower, loa_upper, pct_within_loa), where:
            - bias: mean of (predicted - measured).
            - loa_lower / loa_upper: 95% limits of agreement (bias +/- 1.96*SD).
            - pct_within_loa: percentage of points falling within the LoA.
    """
    diff = np.asarray(y_pred) - np.asarray(y_true)
    bias = diff.mean()
    sd = diff.std()
    loa_lower = bias - 1.96 * sd
    loa_upper = bias + 1.96 * sd
    pct_within = np.mean((diff >= loa_lower) & (diff <= loa_upper)) * 100
    return bias, loa_lower, loa_upper, pct_within


# ══════════════════════════════════════════════════════════════════════════
# MODEL FACTORY & HYPERPARAMETER SEARCH SPACES
# ══════════════════════════════════════════════════════════════════════════

def build_model(algo_name: str, params: dict):
    """Instantiate a regressor with a fixed (already-optimised) parameter set.

    Args:
        algo_name: One of the algorithm names listed in `SUPPORTED_ALGORITHMS`.
        params: Dictionary of hyperparameters, typically `study.best_params`
            from a completed Optuna study (see `optuna_objective`).

    Returns:
        An unfitted scikit-learn / XGBoost regressor instance.
    """
    if algo_name == "Linear Regression":
        return LinearRegression(**params)
    if algo_name == "ElasticNet":
        return ElasticNet(**params, random_state=RANDOM_SEED, max_iter=5000)
    if algo_name == "Random Forest":
        return RandomForestRegressor(**params, random_state=RANDOM_SEED)
    if algo_name == "Extra Trees":
        return ExtraTreesRegressor(**params, random_state=RANDOM_SEED)
    if algo_name == "Gradient Boosting":
        return GradientBoostingRegressor(**params, random_state=RANDOM_SEED, loss="absolute_error")
    if algo_name == "XGBoost":
        return XGBRegressor(**params, random_state=RANDOM_SEED, objective="reg:absoluteerror")
    if algo_name == "Support Vector Regressor (SVR)":
        return SVR(**params)
    if algo_name == "Decision Tree":
        return DecisionTreeRegressor(**params, random_state=RANDOM_SEED)
    if algo_name == "K-Neighbors (KNN)":
        return KNeighborsRegressor(**params)
    if algo_name == "AdaBoost":
        return AdaBoostRegressor(**params, random_state=RANDOM_SEED)
    if algo_name == "Gaussian Process":
        return GaussianProcessRegressor(
            kernel=C(params.get("constant_value", 1.0))
            * Matern(length_scale=params.get("length_scale", 1.0), nu=1.5)
            + WhiteKernel(noise_level=params.get("noise_level", 0.1)),
            alpha=params.get("alpha", 1e-10),
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=RANDOM_SEED,
        )
    if algo_name == "Bayesian Ridge":
        return BayesianRidge(
            alpha_1=params.get("alpha_1", 1e-6),
            lambda_1=params.get("lambda_1", 1e-6),
        )
    if algo_name == "Huber Regressor":
        return HuberRegressor(
            epsilon=params.get("epsilon", 1.35),
            alpha=params.get("alpha", 0.0001),
            max_iter=300,
        )
    raise ValueError(f"Unsupported algorithm: '{algo_name}'.")


def optuna_objective(trial: optuna.Trial, algo_name: str, X_train: np.ndarray, y_train: np.ndarray) -> float:
    """Optuna objective: mean 3-fold inner-CV MAE for a candidate hyperparameter set.

    Builds a model with trial-suggested hyperparameters, evaluates it with a
    3-fold inner cross-validation on the current outer-fold training data
    (imputation + scaling refit within each inner fold), and returns the mean
    MAE across inner folds for Optuna to minimise.

    Args:
        trial: The active Optuna trial, used to sample hyperparameters.
        algo_name: One of the algorithm names listed in `SUPPORTED_ALGORITHMS`.
        X_train: Outer-fold training feature matrix.
        y_train: Outer-fold training target vector.

    Returns:
        Mean absolute error averaged across the 3 inner folds.
    """
    if algo_name == "Linear Regression":
        model = LinearRegression(fit_intercept=trial.suggest_categorical("fit_intercept", [True, False]))
    elif algo_name == "ElasticNet":
        model = ElasticNet(
            alpha=trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
            random_state=RANDOM_SEED, max_iter=5000,
        )
    elif algo_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            random_state=RANDOM_SEED, n_jobs=1,
        )
    elif algo_name == "Extra Trees":
        model = ExtraTreesRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            random_state=RANDOM_SEED, n_jobs=1,
        )
    elif algo_name == "Gradient Boosting":
        model = GradientBoostingRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            random_state=RANDOM_SEED, loss="absolute_error",
        )
    elif algo_name == "XGBoost":
        model = XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            random_state=RANDOM_SEED, objective="reg:absoluteerror", n_jobs=1,
        )
    elif algo_name == "Support Vector Regressor (SVR)":
        model = SVR(
            C=trial.suggest_float("C", 1e-2, 1e2, log=True),
            epsilon=trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
            kernel=trial.suggest_categorical("kernel", ["linear", "rbf"]),
        )
    elif algo_name == "Decision Tree":
        model = DecisionTreeRegressor(
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            random_state=RANDOM_SEED,
        )
    elif algo_name == "K-Neighbors (KNN)":
        model = KNeighborsRegressor(
            n_neighbors=trial.suggest_int("n_neighbors", 2, 15),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
        )
    elif algo_name == "AdaBoost":
        model = AdaBoostRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 2.0, log=True),
            random_state=RANDOM_SEED,
        )
    elif algo_name == "Gaussian Process":
        model = GaussianProcessRegressor(
            kernel=C(trial.suggest_float("constant_value", 0.1, 10.0, log=True))
            * Matern(length_scale=trial.suggest_float("length_scale", 0.1, 10.0, log=True), nu=1.5)
            + WhiteKernel(noise_level=trial.suggest_float("noise_level", 1e-5, 1.0, log=True)),
            alpha=trial.suggest_float("alpha", 1e-10, 1e-2, log=True),
            n_restarts_optimizer=3, normalize_y=True, random_state=RANDOM_SEED,
        )
    elif algo_name == "Bayesian Ridge":
        model = BayesianRidge(
            alpha_1=trial.suggest_float("alpha_1", 1e-6, 1e-2, log=True),
            lambda_1=trial.suggest_float("lambda_1", 1e-6, 1e-2, log=True),
        )
    elif algo_name == "Huber Regressor":
        model = HuberRegressor(
            epsilon=trial.suggest_float("epsilon", 1.1, 2.0),
            alpha=trial.suggest_float("alpha", 1e-5, 0.1, log=True),
            max_iter=300,
        )
    else:
        raise ValueError(f"Unsupported algorithm: '{algo_name}'.")

    inner_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    maes = []
    for tr_idx, val_idx in inner_cv.split(X_train):
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", get_scaler(NORMALIZATION_METHOD)),
            ("regressor", clone(model)),
        ])
        pipeline.fit(X_train[tr_idx], y_train[tr_idx])
        preds = pipeline.predict(X_train[val_idx])
        maes.append(mean_absolute_error(y_train[val_idx], preds))
    return float(np.mean(maes))


# ══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════

def wrap_feature_name(name: str, width: int = 35) -> str:
    """Wrap a long feature name across multiple lines for legible plot axes.

    Args:
        name: Raw feature column name.
        width: Maximum character width per line.

    Returns:
        The feature name with underscores replaced by spaces and line
        breaks inserted at natural word boundaries.
    """
    name_clean = name.replace("_", " ").replace("IBSI:", "IBSI: ")
    return textwrap.fill(name_clean, width=width, break_long_words=False)


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    method_name: str,
    output_dir: str,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    error_margin: float = ERROR_CONE_MARGIN,
) -> None:
    """Save a predicted-vs-actual scatter plot with the identity line and error cone.

    A shaded error cone of +/-`error_margin` (relative to the identity line,
    y = x) is drawn to visually flag predictions falling outside a
    clinically acceptable relative deviation. If `y_lower`/`y_upper` are
    provided (Jackknife+ prediction interval bounds), points are drawn with
    asymmetric error bars instead of plain markers.

    Args:
        y_true: Ground-truth (measured) absorbed doses.
        y_pred: Model-predicted absorbed doses.
        model_name: Algorithm name, used in the title and output filename.
        method_name: Dose-integration method ("Exponential" or "Trapezoid"),
            used in the title and output filename.
        output_dir: Directory where the PNG file will be saved.
        y_lower: Optional lower bound of the prediction interval per sample.
        y_upper: Optional upper bound of the prediction interval per sample.
        error_margin: Half-width of the shaded error cone (0.20 -> +/-20%).
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    lims = [min(np.min(y_true_arr), np.min(y_pred_arr)) * 0.9,
            max(np.max(y_true_arr), np.max(y_pred_arr)) * 1.1]
    x_line = np.linspace(max(lims[0], 0), lims[1], 200)

    # Shaded +/- error_margin cone around the identity line
    ax.fill_between(
        x_line,
        x_line * (1 - error_margin),
        x_line * (1 + error_margin),
        color="steelblue", alpha=0.15,
        label=f"\u00b1{int(error_margin * 100)}% error cone",
    )

    if y_lower is not None and y_upper is not None:
        ax.errorbar(
            y_true_arr, y_pred_arr,
            yerr=[y_pred_arr - y_lower, y_upper - y_pred_arr],
            fmt="o", alpha=0.6, ecolor="steelblue", capsize=3,
            markersize=4, markeredgecolor="k", markeredgewidth=0.4,
            label="Prediction interval (90%)",
        )
    else:
        ax.scatter(y_true_arr, y_pred_arr, alpha=0.7, s=40, edgecolors="k", linewidths=0.4)

    ax.plot(lims, lims, "r--", linewidth=1, label="Identity line (y=x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    r2_val = r2_score(y_true_arr, y_pred_arr)
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    pearson_r, _ = pearsonr(y_true_arr, y_pred_arr)
    ccc = concordance_correlation_coefficient(y_true_arr, y_pred_arr)

    ax.set_xlabel("Actual Dose (Gy/GBq)", fontsize=10)
    ax.set_ylabel("Predicted Dose (Gy/GBq)", fontsize=10)
    ax.set_title(
        f"{model_name}\n{method_name} | R\u00b2={r2_val:.3f} | MAE={mae:.4f} | "
        f"r={pearson_r:.3f} | CCC={ccc:.3f}",
        fontsize=9,
    )
    ax.legend(fontsize=7)
    plt.tight_layout()

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)[:60]
    fig.savefig(os.path.join(output_dir, f"PredVsActual_{method_name}_{safe_name}.png"), dpi=300)
    plt.close(fig)


def plot_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    method_name: str,
    output_dir: str,
) -> None:
    """Save a Bland-Altman agreement plot (fixed axes for cross-model comparison).

    Args:
        y_true: Ground-truth (measured) absorbed doses.
        y_pred: Model-predicted absorbed doses.
        model_name: Algorithm name, used in the title and output filename.
        method_name: Dose-integration method, used in the title and filename.
        output_dir: Directory where the PNG file will be saved.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mean_val = (y_true + y_pred) / 2
    diff = y_pred - y_true
    bias = diff.mean()
    loa = 1.96 * diff.std(ddof=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(mean_val, diff, alpha=0.7, s=40, edgecolors="k", linewidths=0.4)
    ax.axhline(bias, color="red", linestyle="--", linewidth=1, label=f"Bias = {bias:.4f}")
    ax.axhline(bias + loa, color="blue", linestyle=":", linewidth=1, label=f"+1.96 SD = {bias + loa:.4f}")
    ax.axhline(bias - loa, color="blue", linestyle=":", linewidth=1, label=f"-1.96 SD = {bias - loa:.4f}")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)

    # Fixed axes enable direct visual comparison across feature-set variants
    # (e.g. PET+CT vs. PET+CT+CBs) evaluated on the same VOI/dose method.
    # Adjust or remove these limits if your target dose range differs.
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(-0.6, 0.3)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(True, which="major", linestyle="-", linewidth=0.3, alpha=0.4)

    ax.set_xlabel("Mean of Actual and Predicted (Gy/GBq)", fontsize=10)
    ax.set_ylabel("Predicted - Actual (Gy/GBq)", fontsize=10)
    ax.set_title(f"Bland-Altman: {model_name}\n{method_name}", fontsize=9)
    ax.legend(fontsize=7)

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)[:60]
    output_path = os.path.join(output_dir, f"BlandAltman_{method_name}_{safe_name}.png")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_shap(
    model,
    model_name: str,
    X_train_df: pd.DataFrame,
    method_name: str,
    output_dir: str,
    max_display: int = SHAP_MAX_DISPLAY,
) -> None:
    """Save a SHAP summary plot for the fitted model, using the appropriate explainer.

    Uses `shap.TreeExplainer` for tree-based ensembles, `shap.LinearExplainer`
    for linear models, and falls back to a sampled `shap.KernelExplainer` for
    all other estimators (e.g. Bayesian Ridge, Huber, SVR, Gaussian Process).

    Args:
        model: A fitted regressor (the final pipeline step, not the pipeline itself).
        model_name: Algorithm name, used in the title and output filename.
        X_train_df: Feature matrix (already imputed/scaled) as a DataFrame,
            with the fitted model's expected feature columns.
        method_name: Dose-integration method, used in the title and filename.
        output_dir: Directory where the PNG file will be saved.
        max_display: Maximum number of features shown on the summary plot.
    """
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)[:60]
    try:
        tree_models = (
            RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
            AdaBoostRegressor, DecisionTreeRegressor, XGBRegressor,
        )
        linear_models = (LinearRegression, ElasticNet)

        if isinstance(model, tree_models):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train_df)
        elif isinstance(model, linear_models):
            explainer = shap.LinearExplainer(model, X_train_df)
            shap_values = explainer.shap_values(X_train_df)
        else:
            background = shap.sample(X_train_df, min(50, len(X_train_df)), random_state=RANDOM_SEED)
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_train_df, nsamples=100)

        X_display = X_train_df.copy()
        X_display.columns = [wrap_feature_name(c, width=35) for c in X_display.columns]

        n_features_shown = min(max_display, X_display.shape[1])
        fig_height = max(5, 0.6 * n_features_shown)

        fig = plt.figure(figsize=(12, fig_height))
        shap.summary_plot(shap_values, X_display, show=False, plot_size=None, max_display=max_display)
        plt.title(f"SHAP Summary: {model_name} | {method_name}", fontsize=11, pad=15)
        plt.yticks(fontsize=8)
        plt.xlabel("SHAP value (impact on model output)", fontsize=10)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"SHAP_{method_name}_{safe_name}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 -- SHAP explainers can fail for many model-specific reasons
        print(f"      SHAP failed for {model_name}: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# JACKKNIFE+ CONFORMAL PREDICTION
# ══════════════════════════════════════════════════════════════════════════

def jackknife_plus_manual(
    best_params: dict,
    algo_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    alpha: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Jackknife+ leave-one-out predictions and conformal intervals.

    For each held-out sample, fits the model on all remaining samples,
    predicts the held-out point, and records the absolute residual. The
    (1 - alpha) quantile of all residuals is then used as a symmetric,
    distribution-free margin around every leave-one-out prediction.

    Args:
        best_params: Hyperparameters for `algo_name`, typically the best
            parameters found during the outer-fold Optuna search.
        algo_name: One of the algorithm names listed in `SUPPORTED_ALGORITHMS`.
        X: Full feature matrix (all patients).
        y: Full target vector (all patients).
        feature_names: Column names of `X` (unused internally, kept for
            interface consistency with the calling function).
        alpha: Miscoverage rate (e.g. 0.10 for a nominal 90% interval).

    Returns:
        A tuple (y_pred_loo, y_lower, y_upper): the leave-one-out
        predictions and their lower/upper conformal interval bounds.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)
    loo = LeaveOneOut()
    y_pred_loo = np.zeros(n)
    residuals = np.zeros(n)

    for train_idx, test_idx in loo.split(X):
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", get_scaler(NORMALIZATION_METHOD)),
            ("regressor", build_model(algo_name, best_params)),
        ])
        pipeline.fit(X[train_idx], y[train_idx])
        y_hat = pipeline.predict(X[test_idx])[0]
        i = test_idx[0]
        y_pred_loo[i] = y_hat
        residuals[i] = abs(y[i] - y_hat)

    q = np.quantile(residuals, 1 - alpha)
    return y_pred_loo, y_pred_loo - q, y_pred_loo + q


def run_jackknife_plus(
    best_params: dict,
    best_algo_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    method_name: str,
    plots_pred_dir: str,
    plots_bland_dir: str,
    plots_shap_dir: str,
    predictions_dir: str,
) -> dict:
    """Run Jackknife+ for the best outer-loop model and export plots/CSVs.

    Args:
        best_params: Hyperparameters of the best-performing algorithm
            (lowest LOPOCV MAE) across the outer-loop benchmark.
        best_algo_name: Name of the best-performing algorithm.
        X: Full feature matrix (all patients).
        y: Full target vector (all patients).
        feature_names: Column names of `X`.
        method_name: Dose-integration method ("Exponential" or "Trapezoid").
        plots_pred_dir: Output directory for predicted-vs-actual plots.
        plots_bland_dir: Output directory for Bland-Altman plots.
        plots_shap_dir: Output directory for SHAP summary plots.
        predictions_dir: Output directory for per-patient prediction CSVs.

    Returns:
        A dictionary summarising Jackknife+ coverage, interval width, and
        the full set of agreement metrics for the best model.
    """
    print(f"\n  Running manual Jackknife+ for best Optuna model: {best_algo_name}...")

    y_pred_jk, y_lower, y_upper = jackknife_plus_manual(
        best_params, best_algo_name, X, y, feature_names, alpha=JACKKNIFE_ALPHA
    )

    coverage = np.mean((y >= y_lower) & (y <= y_upper))
    interval_width = np.mean(y_upper - y_lower)
    mae_jk = mean_absolute_error(y, y_pred_jk)
    r2_jk = r2_score(y, y_pred_jk)
    pearson_r_jk, _ = pearsonr(y, y_pred_jk)

    ccc_jk = concordance_correlation_coefficient(y, y_pred_jk)
    icc_jk = intraclass_correlation_coefficient(y, y_pred_jk)
    bias_jk, loa_lo, loa_hi, pct_jk = bland_altman_stats(y, y_pred_jk)

    print(f"    Coverage ({int((1 - JACKKNIFE_ALPHA) * 100)}% target): {coverage:.1%}")
    print(f"    Mean interval width:   {interval_width:.4f} Gy/GBq")
    print(f"    MAE:                   {mae_jk:.4f} Gy/GBq")
    print(f"    R\u00b2:                    {r2_jk:.3f}")
    print(f"    CCC:                   {ccc_jk:.3f}")
    print(f"    ICC(2,1):              {icc_jk:.3f}")
    print(f"    Bland-Altman Bias:     {bias_jk:.4f} Gy/GBq")
    print(f"    LoA: [{loa_lo:.4f}, {loa_hi:.4f}]  ({pct_jk:.1f}% within)")

    plot_predicted_vs_actual(
        y, y_pred_jk, f"{best_algo_name} (Jackknife+ manual)", method_name,
        plots_pred_dir, y_lower=y_lower, y_upper=y_upper,
    )
    plot_bland_altman(y, y_pred_jk, f"{best_algo_name} (Jackknife+ manual)", method_name, plots_bland_dir)

    p_full = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", get_scaler(NORMALIZATION_METHOD)),
        ("regressor", build_model(best_algo_name, best_params)),
    ])
    p_full.fit(X, y)
    X_transformed_df = pd.DataFrame(p_full[:-1].transform(X), columns=feature_names)
    plot_shap(p_full.named_steps["regressor"], f"{best_algo_name} (Jackknife+)", X_transformed_df, method_name, plots_shap_dir)

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in best_algo_name)[:60]
    jk_df = pd.DataFrame({
        "y_true": y,
        "y_pred": y_pred_jk,
        "lower_90": y_lower,
        "upper_90": y_upper,
        "interval_width": y_upper - y_lower,
        "within_interval": (y >= y_lower) & (y <= y_upper),
    })
    jk_df.to_csv(os.path.join(predictions_dir, f"JackknifePlus_{method_name}_{safe_name}.csv"), index=False)
    print("    Jackknife+ results saved.")

    return {
        "Dose_Method": method_name,
        "Best_Algorithm": best_algo_name,
        "Coverage_90pct": coverage,
        "Mean_Interval_Width": interval_width,
        "MAE_(Gy/GBq)": mae_jk,
        "R2_Score": r2_jk,
        "Pearson_r": pearson_r_jk,
        "CCC": ccc_jk,
        "ICC(2,1)": icc_jk,
        "BA_Bias_(Gy/GBq)": bias_jk,
        "BA_LoA_Lower": loa_lo,
        "BA_LoA_Upper": loa_hi,
        "BA_Pct_Within_LoA": pct_jk,
    }


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════

def load_and_prepare_dataset(file_path: str, method_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load a LASSO/BORUTA-selected feature CSV and prepare it for modelling.

    Cleans and coerces the target column to numeric, drops rows with a
    missing target, drops non-feature columns (patient/ROI identifiers and
    both dose-method targets), and sanitises feature column names for
    downstream estimator compatibility (e.g. XGBoost rejects some special
    characters in column names).

    Args:
        file_path: Path to the input CSV.
        method_name: Dose-integration method whose target column should be
            used ("Exponential" or "Trapezoid").

    Returns:
        A tuple (X, y, groups, feature_names):
            - X: Feature matrix (float array), rows with a missing target removed.
            - y: Target vector (absorbed dose, Gy/GBq).
            - groups: Patient ID per row, for group-aware cross-validation.
            - feature_names: Sanitised feature column names, in `X` column order.
    """
    df = pd.read_csv(file_path)
    target_col = f"Absorbed Dose per Injected Activity (Gy/GBq) {method_name}"

    y_raw = pd.to_numeric(df[target_col].astype(str).str.replace(",", "."), errors="coerce")
    nan_count = y_raw.isna().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values in target column — removing these rows.")
    valid_mask = y_raw.notna()
    y_raw = y_raw[valid_mask].reset_index(drop=True)
    df = df[valid_mask].reset_index(drop=True)

    groups = df["Patient"].values

    cols_to_drop = [
        "Patient", "ROI",
        "Absorbed Dose per Injected Activity (Gy/GBq) Exponential",
        "Absorbed Dose per Injected Activity (Gy/GBq) Trapezoid",
    ]
    X_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    for col in X_df.columns:
        if X_df[col].dtype == "object":
            X_df[col] = pd.to_numeric(X_df[col].astype(str).str.replace(",", "."), errors="coerce")

    X_df.columns = (
        X_df.columns
        .str.replace("[", "_", regex=False).str.replace("]", "_", regex=False)
        .str.replace("<", "_", regex=False).str.replace(">", "_", regex=False)
        .str.replace("(", "_", regex=False).str.replace(")", "_", regex=False)
    )

    return X_df.values, y_raw.values, groups, list(X_df.columns)


def validate_algorithms(algorithms: list[str]) -> None:
    """Raise early if `ALGORITHMS` contains an unrecognised name.

    Args:
        algorithms: The user-configured list of algorithm names.

    Raises:
        ValueError: If any entry is not in `SUPPORTED_ALGORITHMS`.
    """
    unknown = [a for a in algorithms if a not in SUPPORTED_ALGORITHMS]
    if unknown:
        raise ValueError(
            f"Unsupported algorithm(s) in ALGORITHMS: {unknown}. "
            f"Supported options are: {sorted(SUPPORTED_ALGORITHMS)}"
        )


# ══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the full benchmarking pipeline for the configured VOI.

    For every dose-integration method in `DATASET_PATHS`, benchmarks every
    algorithm in `ALGORITHMS` via LOPOCV + Optuna hyperparameter search,
    exports per-model plots/CSVs, and finally runs Jackknife+ on the
    best-performing model per method.
    """
    validate_algorithms(ALGORITHMS)

    print("INITIATING BAYESIAN OPTIMISATION PIPELINE (OPTUNA + LOPOCV + JACKKNIFE+)...")
    print(f"VOI: {VOI_NAME} | Normalisation: {NORMALIZATION_METHOD} | Algorithms: {ALGORITHMS}")
    print(f"Predicted-vs-actual error cone: \u00b1{int(ERROR_CONE_MARGIN * 100)}%")
    print("Warning: this process is computationally intensive.")

    all_results = []
    jackknife_results = []

    for method_name, file_path in DATASET_PATHS.items():
        if not os.path.exists(file_path):
            print(f"\nFile {file_path} not found. Skipping...")
            continue

        print("\n" + "=" * 80)
        print(f"BAYESIAN TUNING FOR: {VOI_NAME.upper()} | {method_name.upper()} DOSE")
        print("=" * 80)

        plots_pred_vs_actual = os.path.join(OUTPUT_ROOT, method_name, "PredVsActual")
        plots_bland = os.path.join(OUTPUT_ROOT, method_name, "BlandAltman")
        plots_shap = os.path.join(OUTPUT_ROOT, method_name, "SHAP")
        predictions_dir = os.path.join(OUTPUT_ROOT, method_name, "Predictions")
        for d in (plots_pred_vs_actual, plots_bland, plots_shap, predictions_dir):
            os.makedirs(d, exist_ok=True)

        X, y, groups, feature_names = load_and_prepare_dataset(file_path, method_name)

        logo = LeaveOneGroupOut()
        best_mae = np.inf
        best_algo_name: Optional[str] = None
        best_params_global: Optional[dict] = None

        for algo_name in ALGORITHMS:
            print(f"  Optimizing {algo_name}...")

            y_true_all, y_pred_all, best_params_per_fold = [], [], []

            for train_idx, test_idx in logo.split(X, y, groups):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
                study.optimize(lambda trial: optuna_objective(trial, algo_name, X_train, y_train), n_trials=N_TRIALS)

                best_params = study.best_params
                best_params_per_fold.append(best_params)

                p_final = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", get_scaler(NORMALIZATION_METHOD)),
                    ("regressor", build_model(algo_name, best_params)),
                ])
                p_final.fit(X_train, y_train)
                pred = p_final.predict(X_test)

                y_true_all.extend(y_test)
                y_pred_all.extend(pred)

            y_true_arr, y_pred_arr = np.array(y_true_all), np.array(y_pred_all)

            r2 = r2_score(y_true_arr, y_pred_arr)
            mae = mean_absolute_error(y_true_arr, y_pred_arr)
            rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
            mape = mean_absolute_percentage_error(y_true_arr, y_pred_arr)
            pearson_r, _ = pearsonr(y_true_arr, y_pred_arr)
            ccc = concordance_correlation_coefficient(y_true_arr, y_pred_arr)
            icc = intraclass_correlation_coefficient(y_true_arr, y_pred_arr)
            ba_bias, ba_lo, ba_hi, ba_pct = bland_altman_stats(y_true_arr, y_pred_arr)

            all_results.append({
                "VOI": VOI_NAME,
                "Dose_Method": method_name,
                "Algorithm": algo_name,
                "R2_Score": r2,
                "Pearson_r": pearson_r,
                "MAE_(Gy/GBq)": mae,
                "RMSE_(Gy/GBq)": rmse,
                "MAPE": mape,
                "CCC": ccc,
                "ICC(2,1)": icc,
                "BA_Bias_(Gy/GBq)": ba_bias,
                "BA_LoA_Lower": ba_lo,
                "BA_LoA_Upper": ba_hi,
                "BA_Pct_Within_LoA": ba_pct,
            })

            print(f"    {algo_name:<35} | MAE: {mae:.4f} | R\u00b2: {r2:+.3f} | CCC: {ccc:.3f} | ICC: {icc:.3f}")

            if mae < best_mae:
                best_mae, best_algo_name, best_params_global = mae, algo_name, best_params_per_fold[-1]

            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in algo_name)[:60]
            pd.DataFrame({
                "Patient": groups, "y_true": y, "y_pred": y_pred_arr, "residual": y_pred_arr - y,
            }).to_csv(os.path.join(predictions_dir, f"Predictions_{method_name}_{safe_name}.csv"), index=False)

            plot_predicted_vs_actual(y_true_arr, y_pred_arr, algo_name, method_name, plots_pred_vs_actual)
            plot_bland_altman(y_true_arr, y_pred_arr, algo_name, method_name, plots_bland)

            p_shap = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", get_scaler(NORMALIZATION_METHOD)),
                ("regressor", build_model(algo_name, best_params_per_fold[-1])),
            ])
            p_shap.fit(X, y)
            X_transformed_df = pd.DataFrame(p_shap[:-1].transform(X), columns=feature_names)
            plot_shap(p_shap.named_steps["regressor"], algo_name, X_transformed_df, method_name, plots_shap)

        if best_algo_name is not None:
            jk_summary = run_jackknife_plus(
                best_params_global, best_algo_name, X, y, feature_names, method_name,
                plots_pred_vs_actual, plots_bland, plots_shap, predictions_dir,
            )
            jk_summary["VOI"] = VOI_NAME
            jackknife_results.append(jk_summary)

    df_results = pd.DataFrame(all_results).sort_values(by=["Dose_Method", "MAE_(Gy/GBq)"], ascending=[True, True])
    df_results.to_csv(f"ML_LASSO_OPTUNA_{VOI_NAME}_Performance_Comparison.csv", index=False)

    if jackknife_results:
        pd.DataFrame(jackknife_results).to_csv(f"ML_LASSO_OPTUNA_{VOI_NAME}_JackknifePlus_Summary.csv", index=False)

    print("\n" + "=" * 80)
    print("BAYESIAN OPTIMISATION + JACKKNIFE+ COMPLETE.")
    print(f"Performance table : ML_LASSO_OPTUNA_{VOI_NAME}_Performance_Comparison.csv")
    print(f"Jackknife+ summary: ML_LASSO_OPTUNA_{VOI_NAME}_JackknifePlus_Summary.csv")
    print("\nTop Performing Models per Dose Method (Lowest MAE):")
    for method in df_results["Dose_Method"].unique():
        best = df_results[df_results["Dose_Method"] == method].iloc[0]
        print(
            f"  -> {method}: {best['Algorithm']} "
            f"(MAE: {best['MAE_(Gy/GBq)']:.4f} | R\u00b2: {best['R2_Score']:.3f} | "
            f"CCC: {best['CCC']:.3f} | ICC: {best['ICC(2,1)']:.3f})"
        )


if __name__ == "__main__":
    main()
