# IBM HR Attrition: Executive-Level Predictive Turnover Report

This project trains a scikit-learn model on IBM's HR dataset to predict employee attrition (`Attrition = Yes/No`) and identify the most stable drivers of turnover for executive reporting.

## Stack
- Python
- pandas
- scikit-learn

## Project Files
- `train_attrition_model.py` - training/evaluation pipeline + driver analysis
- `IBM-HR-Employee-Attrition-Performance.xlsx` - source dataset
- `artifacts/` - generated model and reports after training

## How to Run
```bash
pip install -r requirements.txt
python train_attrition_model.py
python retention_roi.py
```

## Outputs
After running, the script generates:
- `artifacts/attrition_model.joblib` - trained model pipeline
- `artifacts/cv_metrics.csv` - fold-level metrics across repeated CV
- `artifacts/metrics.json` - cross-validated metric summary (mean/std)
- `artifacts/feature_importance_folds.csv` - feature importance per fold
- `artifacts/feature_importance.csv` - stability table by feature (mean importance, mean rank, top-3 frequency)
- `artifacts/top_3_drivers.csv` - stable top 3 turnover drivers
- `artifacts/executive_summary.md` - leadership-ready summary and interpretation notes
- `artifacts/retention_roi_scenarios.csv` - low/base/high savings scenarios by intervention
- `artifacts/roi_summary.md` - concise ROI narrative for interview/executive discussion
- `artifacts/executive_summary.md` - now auto-appended with an Executive Q&A section from ROI scenarios

## Modeling Approach
- Target encoding: `Yes -> 1`, `No -> 0`
- Data governance: drops non-actionable fields (`EmployeeNumber`) and constant fields (`EmployeeCount`, `Over18`, `StandardHours`)
- Preprocessing:
  - Numeric features: median imputation
  - Categorical features: most-frequent imputation + one-hot encoding
- Model: `RandomForestClassifier` with `class_weight='balanced'`
- Validation: repeated stratified cross-validation (5 folds x 10 repeats = 50 out-of-sample tests)
- Driver detection: tree-based feature importance aggregated back to original feature names
- Stability rule: top drivers prioritized by frequency of appearing in the fold-level top 3

## Executive Interpretation Guardrails
- Treat results as predictive signals, not causal proof.
- Prioritize interventions where a driver is both high-importance and stable across folds.
- Re-run routinely (e.g., quarterly) to track drift in employee risk patterns.

## Retention ROI Layer
- `retention_roi.py` scores model-based expected leavers and estimates financial impact from interventions on top drivers.
- Segment targeting is data-driven:
  - Numeric drivers: highest-risk quartile
  - Categorical drivers: highest-risk category
- Assumptions are configurable in `roi_assumptions.json`:
  - Cost per attrition (`low/base/high`)
  - Intervention coverage
  - Effect size (relative risk reduction)
  - Program fixed and variable costs
