#!/usr/bin/env python3
"""
Overfitting Analysis Script
============================
Checks if Linear Regression is overfitting by comparing train vs test performance.
"""

import os
import sys
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Setup paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(project_root, 'data', 'processed')
MODELS = os.path.join(project_root, 'models')

print("="*70)
print("LINEAR REGRESSION OVERFITTING ANALYSIS")
print("="*70)

# 1. Load data
print("\n1. LOADING DATA...")
df = pd.read_csv(os.path.join(DATA_PROCESSED, 'ml_features.csv'), index_col=0, parse_dates=True)
print(f"   Total samples: {len(df)}")
print(f"   Features: {len(df.columns) - 1}")

# 2. Prepare features and target
X = df.drop('target', axis=1)
y = df['target']

# 3. Split (same as training)
split_idx = int(len(df) * 0.8)
X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

print(f"\n2. DATA SPLIT...")
print(f"   Train: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"   Test: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")

# 4. Load model and scaler
print("\n3. LOADING MODEL...")
model = joblib.load(os.path.join(MODELS, 'best_ml_model_linear_regression.joblib'))
scaler = joblib.load(os.path.join(MODELS, 'ml_scaler.joblib'))

# 5. Scale data (same as training)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Make predictions
print("\n4. MAKING PREDICTIONS...")
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# 7. Calculate metrics
print("\n5. CALCULATING METRICS...")
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

# 8. Display results
print("\n" + "="*70)
print("PERFORMANCE COMPARISON")
print("="*70)
print(f"\n{'Metric':<15} {'Train':<15} {'Test':<15} {'Difference':<15}")
print("-" * 70)
print(f"{'MAE':<15} {train_mae:<15.6f} {test_mae:<15.6f} {abs(train_mae - test_mae):<15.6f}")
print(f"{'RMSE':<15} {train_rmse:<15.6f} {test_rmse:<15.6f} {abs(train_rmse - test_rmse):<15.6f}")
print(f"{'R²':<15} {train_r2:<15.6f} {test_r2:<15.6f} {abs(train_r2 - test_r2):<15.6f}")

# 9. Overfitting diagnosis
print("\n" + "="*70)
print("OVERFITTING DIAGNOSIS")
print("="*70)

r2_diff = train_r2 - test_r2
mae_ratio = test_mae / train_mae

print(f"\nR² Difference (train - test): {r2_diff:.6f}")
print(f"MAE Ratio (test / train): {mae_ratio:.4f}")

if r2_diff < 0.05 and 0.9 < mae_ratio < 1.1:
    print("\n✅ GOOD GENERALIZATION - No significant overfitting detected!")
    print("   - R² difference is small (< 0.05)")
    print("   - Test and train errors are similar")
    print("   - Model generalizes well to unseen data")
elif r2_diff < 0.1:
    print("\n⚠️  MILD OVERFITTING - Some overfitting detected")
    print("   - R² difference is moderate (0.05-0.10)")
    print("   - Model may benefit from regularization")
else:
    print("\n❌ SIGNIFICANT OVERFITTING - Model is overfitting!")
    print("   - R² difference is large (> 0.10)")
    print("   - Model memorizing training data")
    print("   - Regularization strongly recommended")

# 10. Additional checks
print("\n" + "="*70)
print("ADDITIONAL CHECKS")
print("="*70)

# Check residuals
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

print(f"\nResidual Statistics:")
print(f"   Train residuals mean: {train_residuals.mean():.6f} (should be ~0)")
print(f"   Test residuals mean: {test_residuals.mean():.6f} (should be ~0)")
print(f"   Train residuals std: {train_residuals.std():.6f}")
print(f"   Test residuals std: {test_residuals.std():.6f}")

# Check for systematic bias
if abs(test_residuals.mean()) > 0.001:
    print("\n⚠️  WARNING: Test residuals have non-zero mean")
    print("   This suggests systematic bias in predictions")

# Feature importance (top 10)
print(f"\n" + "="*70)
print("TOP 10 MOST IMPORTANT FEATURES")
print("="*70)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_
})
feature_importance['abs_coefficient'] = abs(feature_importance['coefficient'])
feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)

print(f"\n{'Feature':<20} {'Coefficient':<15} {'Impact':<10}")
print("-" * 50)
for idx, row in feature_importance.head(10).iterrows():
    impact = "↑ Positive" if row['coefficient'] > 0 else "↓ Negative"
    print(f"{row['feature']:<20} {row['coefficient']:<15.6f} {impact:<10}")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

if r2_diff < 0.05:
    print("\n1. Current model is performing well - no immediate action needed")
    print("2. Consider ensemble methods to potentially improve further")
    print("3. Document why Linear Regression works so well")
else:
    print("\n1. Implement Ridge/Lasso regularization")
    print("2. Perform feature selection to reduce dimensionality")
    print("3. Use cross-validation to tune hyperparameters")
    print("4. Consider simpler models or feature engineering")

print("\n" + "="*70)
