# BlendNet-S IC50 Prediction Performance Analysis

## Overview
This report analyzes the performance of the BlendNet-S model on IC50 prediction tasks across different data splitting strategies and cross-validation folds.

## Average Performance by Split Type

| Split Type   |         R² |     RMSE |    R² Std |   RMSE Std |
|:-------------|-----------:|---------:|----------:|-----------:|
| blind_split  | -0.409584  | 1.56383  | 0.0621019 | 0.00954879 |
| new_compound |  0.412071  | 1.03952  | 0.0155726 | 0.00813268 |
| new_protein  | -0.0748383 | 1.39321  | 0.0415603 | 0.0158605  |
| random_split |  0.712104  | 0.763899 | 0.010708  | 0.0144412  |

## Key Observations

- **Best Performing Split**: random_split with R² = 0.7121 and RMSE = 0.7639
- **Worst Performing Split**: blind_split with R² = -0.4096 and RMSE = 1.5638

### IC50 Value Distributions

#### blind_split
- Mean IC50: 7.2440
- Median IC50: 7.2840
- Standard Deviation: 1.3181
- Range: 2.0000 - 12.0000

#### random_split
- Mean IC50: 6.5821
- Median IC50: 6.5850
- Standard Deviation: 1.4239
- Range: 2.0000 - 12.0000

#### new_protein
- Mean IC50: 6.9178
- Median IC50: 6.9666
- Standard Deviation: 1.3455
- Range: 2.0000 - 12.0000

#### new_compound
- Mean IC50: 7.0349
- Median IC50: 7.1487
- Standard Deviation: 1.3561
- Range: 2.0000 - 12.0000

### Split Type Analysis

#### blind_split
- Average R²: -0.4096 (±0.0621)
- Average RMSE: 1.5638 (±0.0095)
- Best CV Fold: CV1 (R² = -0.3480)
- Worst CV Fold: CV0 (R² = -0.4722)

#### new_compound
- Average R²: 0.4121 (±0.0156)
- Average RMSE: 1.0395 (±0.0081)
- Best CV Fold: CV0 (R² = 0.4286)
- Worst CV Fold: CV1 (R² = 0.3976)

#### new_protein
- Average R²: -0.0748 (±0.0416)
- Average RMSE: 1.3932 (±0.0159)
- Best CV Fold: CV2 (R² = -0.0494)
- Worst CV Fold: CV1 (R² = -0.1228)

#### random_split
- Average R²: 0.7121 (±0.0107)
- Average RMSE: 0.7639 (±0.0144)
- Best CV Fold: CV2 (R² = 0.7238)
- Worst CV Fold: CV0 (R² = 0.7029)

## Summary

The model shows good prediction capability in some splitting strategies, particularly in random splits where data points are randomly assigned to train/test sets. Performance is notably worse in blind_split scenarios, suggesting challenges in generalizing to completely unseen data patterns.

The new_compound split shows moderate performance, indicating some ability to predict properties of novel chemical compounds, while the new_protein split demonstrates difficulties in generalizing to unseen protein targets.

These results highlight the importance of considering the evaluation protocol when assessing model performance, especially in drug discovery applications where generalization to novel compounds or targets is critical.
