# Pre-treatment Dose Prediction in [¹⁷⁷Lu]Lu-PSMA-617 Radioligand Therapy

Machine learning code used in the MSc dissertation *"Pre-treatment
prediction of absorbed dose distributions in patients undergoing
personalized [¹⁷⁷Lu]Lu-PSMA therapy"* (NOVA University Lisbon, Department of
Physics, 2026), to predict organ-at-risk (OAR) and tumour-lesion mean
absorbed doses from pre-therapeutic [⁶⁸Ga]Ga-PSMA-11 PET/CT radiomic
features and clinical biomarkers.

This repository is shared for **transparency and reference**: it shows how
the regression models reported in the dissertation were trained and
evaluated. It is not packaged as a plug-and-play pipeline — no patient data
is included, and reproducing the reported results requires a compatible
dataset that is not publicly available (see disclaimer below).

## Repository Structure

```
.
├── train_ml_models_post_selection.py   # Model training on pre-selected features (Methodology 1)
├── train_lasso_infold_models.py        # LASSO selection repeated in-fold + training (Methodology 2)
├── requirements.txt
└── README.md
```

Two feature-selection protocols are compared in the dissertation (Section
4.4.4) and implemented here as two independent scripts:

- **`train_ml_models_post_selection.py`** — trains and evaluates several
  regression algorithms on a feature set that has already been reduced by
  LASSO or BORUTA feature selection, performed once on the full dataset
  before cross-validation ("Methodology 1").
- **`train_lasso_infold_models.py`** — repeats LASSO feature selection
  independently inside every Leave-One-Patient-Out cross-validation fold,
  using only that fold's training data ("Methodology 2"), a more
  conservative, leakage-free alternative.

Both scripts implement Leave-One-Patient-Out cross-validation (LOPOCV),
Optuna-based Bayesian hyperparameter optimisation, Jackknife+ conformal
prediction intervals, and the same set of agreement metrics (R², Pearson's
r, MAE, RMSE, MAPE, Lin's CCC, ICC(2,1), Bland–Altman bias/LoA).

## Data Disclaimer

This repository does **not** include any patient data or dataset files.
All radiomic and clinical biomarker data used in the dissertation were
collected under an institutional ethics approval at the Nuclear
Medicine–Radiopharmacology Service of the Champalimaud Foundation and are
not publicly distributable. The code is shared purely to document the
modelling methodology; it is not intended to be run end-to-end without
adapting it to a dataset of your own.

## Dependencies

See `requirements.txt` for the Python packages used across both scripts.
Install with:

```bash
pip install -r requirements.txt
```

Tested with Python 3.12.

## Citation

If you reference or adapt this code, please cite:

> Simões, R. T. M. (2026). *Pre-treatment prediction of absorbed dose
> distributions in patients undergoing personalized [¹⁷⁷Lu]Lu-PSMA therapy*
> (Master's dissertation). NOVA School of Science and Technology, NOVA
> University Lisbon.

## License

Specify a license (e.g. MIT) appropriate for your institution's policy
before making this repository public.
