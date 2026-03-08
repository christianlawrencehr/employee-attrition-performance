# Executive Summary: Employee Attrition Risk Model

- Report generated: 2026-03-07 21:56:32
- Sample size: 1470 employees
- Attrition prevalence: 16.1%
- Validation design: Repeated Stratified 5-Fold CV (10 repeats, 50 total folds)

## Model Performance
- ROC-AUC (mean): 0.806 (std: 0.032)
- Accuracy (mean): 0.854
- Precision (mean): 0.795
- Recall (mean): 0.132
- F1 (mean): 0.223

## Stable Top 3 Drivers of Turnover
Drivers are ranked by how often they appear in the top 3 across folds (stability), then by average importance.

- 1. OverTime: in top 3 for 100.0% of folds; avg importance 0.0770
- 2. MonthlyIncome: in top 3 for 100.0% of folds; avg importance 0.0701
- 3. Age: in top 3 for 92.0% of folds; avg importance 0.0572

## Interpretation Guidance
- These are predictive drivers, not causal proof.
- Use drivers to prioritize policy review and targeted retention actions.
- Re-run quarterly to monitor shifts in workforce dynamics.

## Top 10 Stability Table

| Rank | Feature | Top-3 Frequency | Mean Importance | Std Importance |
|---|---|---:|---:|---:|
| 1 | OverTime | 100.0% | 0.0770 | 0.0081 |
| 2 | MonthlyIncome | 100.0% | 0.0701 | 0.0043 |
| 3 | Age | 92.0% | 0.0572 | 0.0037 |
| 4 | DailyRate | 6.0% | 0.0492 | 0.0022 |
| 5 | TotalWorkingYears | 2.0% | 0.0470 | 0.0030 |
| 6 | MonthlyRate | 0.0% | 0.0438 | 0.0016 |
| 7 | YearsAtCompany | 0.0% | 0.0438 | 0.0031 |
| 8 | JobRole | 0.0% | 0.0437 | 0.0020 |
| 9 | HourlyRate | 0.0% | 0.0428 | 0.0015 |
| 10 | DistanceFromHome | 0.0% | 0.0400 | 0.0024 |

## Executive Q&A

### Q: Based on your data, how can we save money by addressing Overtime?
- Base: avoid 9.53 leavers, gross savings $476,680, net savings $350,080, ROI 2.77x
- High: avoid 14.30 leavers, gross savings $1,144,032, net savings $1,017,432, ROI 8.04x
- Low: avoid 4.77 leavers, gross savings $143,004, net savings $16,404, ROI 0.13x

### Q: How much would we save if we reduced overtime by 10%?
- Expected avoided leavers: 11.92
- Gross savings (base cost per attrition): $595,850
- Net savings at 80% coverage (base program assumptions): $350,080
- Net savings at 100% coverage (same assumptions): $452,650
