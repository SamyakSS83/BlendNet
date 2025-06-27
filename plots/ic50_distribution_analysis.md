# IC50 Value Distribution Analysis

## Overview
This report analyzes the distribution of IC50 values across different data splitting strategies in the BlendNet-S model prediction tasks.

## IC50 Value Distributions by Split Type

### blind_split
- **Sample Size**: 381944
- **Mean IC50**: 7.2440
- **Median IC50**: 7.2840
- **Standard Deviation**: 1.3181
- **Range**: 2.0000 - 12.0000
- **Interquartile Range (IQR)**: 6.3215 - 8.1863

#### Actual vs Predicted Distributions
- **Actual Mean**: 7.2440
- **Predicted Mean**: 6.4743
- **Mean Difference**: 0.7698
- **Actual Std Dev**: 1.3181
- **Predicted Std Dev**: 0.6516

### random_split
- **Sample Size**: 502290
- **Mean IC50**: 6.5821
- **Median IC50**: 6.5850
- **Standard Deviation**: 1.4239
- **Range**: 2.0000 - 12.0000
- **Interquartile Range (IQR)**: 5.5045 - 7.6314

#### Actual vs Predicted Distributions
- **Actual Mean**: 6.5821
- **Predicted Mean**: 6.6990
- **Mean Difference**: -0.1169
- **Actual Std Dev**: 1.4239
- **Predicted Std Dev**: 1.1021

### new_protein
- **Sample Size**: 505579
- **Mean IC50**: 6.9178
- **Median IC50**: 6.9666
- **Standard Deviation**: 1.3455
- **Range**: 2.0000 - 12.0000
- **Interquartile Range (IQR)**: 5.9996 - 7.8928

#### Actual vs Predicted Distributions
- **Actual Mean**: 6.9178
- **Predicted Mean**: 6.3847
- **Mean Difference**: 0.5330
- **Actual Std Dev**: 1.3455
- **Predicted Std Dev**: 0.6741

### new_compound
- **Sample Size**: 502641
- **Mean IC50**: 7.0349
- **Median IC50**: 7.1487
- **Standard Deviation**: 1.3561
- **Range**: 2.0000 - 12.0000
- **Interquartile Range (IQR)**: 6.1500 - 8.0000

#### Actual vs Predicted Distributions
- **Actual Mean**: 7.0349
- **Predicted Mean**: 6.9097
- **Mean Difference**: 0.1252
- **Actual Std Dev**: 1.3561
- **Predicted Std Dev**: 0.9067

## Observations and Insights

- The overall mean IC50 value across all split types is 6.9256
- The overall median IC50 value is 7.0000
- The split type with the highest mean IC50 value is **blind_split** (7.2440)
- The split type with the lowest mean IC50 value is **random_split** (6.5821)

### Prediction Biases
- The model under-predicts IC50 values for **blind_split** split by an average of 0.7698 units
- The model under-predicts IC50 values for **new_protein** split by an average of 0.5330 units

## Summary

The analysis of IC50 value distributions across different split types reveals important patterns in both the dataset and model predictions. Understanding these distributions helps to interpret the model's performance metrics and identify potential areas for improvement in the prediction capabilities.