#!/usr/bin/env python3
"""
LSTM Diagnostic Script
======================
This script diagnoses why the LSTM model has extremely negative R² values.
It checks for scaling issues, data leakage, and proper inverse transformation.
"""

import os
import sys
import pickle
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setup paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(project_root, 'data', 'processed')
MODELS = os.path.join(project_root, 'models')

print("="*70)
print("LSTM MODEL DIAGNOSTIC ANALYSIS")
print("="*70)

# 1. Load LSTM data
print("\n1. LOADING LSTM DATA...")
lstm_data_path = os.path.join(DATA_PROCESSED, 'lstm_sequences.pkl')
with open(lstm_data_path, 'rb') as f:
    lstm_data = pickle.load(f)

X_train = lstm_data['X_train']
y_train = lstm_data['y_train']
X_test = lstm_data['X_test']
y_test = lstm_data['y_test']

print(f"   X_train shape: {X_train.shape}")
print(f"   y_train shape: {y_train.shape}")
print(f"   X_test shape: {X_test.shape}")
print(f"   y_test shape: {y_test.shape}")

# 2. Check data ranges (scaled or unscaled?)
print("\n2. CHECKING DATA RANGES...")
print(f"   y_train range: [{y_train.min():.6f}, {y_train.max():.6f}]")
print(f"   y_test range: [{y_test.min():.6f}, {y_test.max():.6f}]")

if y_train.max() <= 1.0 and y_train.min() >= 0.0:
    print("   ✓ Data appears to be SCALED (0-1 range)")
    data_is_scaled = True
else:
    print("   ✓ Data appears to be UNSCALED (raw prices)")
    data_is_scaled = False

# 3. Load scaler
print("\n3. LOADING SCALER...")
scaler_path = os.path.join(MODELS, 'lstm_scaler.joblib')
scaler = joblib.load(scaler_path)
print(f"   Scaler type: {type(scaler).__name__}")
print(f"   Scaler min: {scaler.data_min_}")
print(f"   Scaler max: {scaler.data_max_}")
print(f"   Scaler range: {scaler.data_range_}")

# 4. Load LSTM metrics
print("\n4. LOADING LSTM METRICS...")
metrics_path = os.path.join(MODELS, 'lstm_metrics.json')
with open(metrics_path, 'r') as f:
    lstm_metrics = json.load(f)

print(f"   Reported MAE: {lstm_metrics['mae']:.6f}")
print(f"   Reported R²: {lstm_metrics['r2']:.6f}")

# 5. Load the trained model and make predictions
print("\n5. LOADING MODEL AND MAKING PREDICTIONS...")
try:
    from tensorflow.keras.models import load_model
    
    # Try different model files
    model_files = ['lstm_simple_best.h5', 'lstm_trained_model.keras', 'lstm_best_model.h5']
    model = None
    model_file_used = None
    
    for model_file in model_files:
        model_path = os.path.join(MODELS, model_file)
        if os.path.exists(model_path):
            try:
                model = load_model(model_path)
                model_file_used = model_file
                print(f"   ✓ Loaded model from: {model_file}")
                break
            except Exception as e:
                print(f"   ⚠ Failed to load {model_file}: {str(e)[:50]}")
                continue
    
    if model is None:
        print("   ✗ Could not load any LSTM model!")
        sys.exit(1)
    
    # Make predictions
    y_pred_scaled = model.predict(X_test, verbose=0)
    print(f"   Predictions shape: {y_pred_scaled.shape}")
    print(f"   Predictions range: [{y_pred_scaled.min():.6f}, {y_pred_scaled.max():.6f}]")
    
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    sys.exit(1)

# 6. Calculate metrics in SCALED space
print("\n6. METRICS IN SCALED SPACE...")
mae_scaled = mean_absolute_error(y_test, y_pred_scaled)
rmse_scaled = np.sqrt(mean_squared_error(y_test, y_pred_scaled))
r2_scaled = r2_score(y_test, y_pred_scaled)

print(f"   MAE (scaled): {mae_scaled:.6f}")
print(f"   RMSE (scaled): {rmse_scaled:.6f}")
print(f"   R² (scaled): {r2_scaled:.6f}")

# 7. Inverse transform and calculate metrics in UNSCALED space
print("\n7. INVERSE TRANSFORMING TO ORIGINAL SCALE...")
try:
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_unscaled = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    
    print(f"   Actual (unscaled) range: [{y_test_unscaled.min():.4f}, {y_test_unscaled.max():.4f}]")
    print(f"   Predicted (unscaled) range: [{y_pred_unscaled.min():.4f}, {y_pred_unscaled.max():.4f}]")
    
    mae_unscaled = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
    rmse_unscaled = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
    r2_unscaled = r2_score(y_test_unscaled, y_pred_unscaled)
    
    print(f"\n   MAE (unscaled): ${mae_unscaled:.4f}")
    print(f"   RMSE (unscaled): ${rmse_unscaled:.4f}")
    print(f"   R² (unscaled): {r2_unscaled:.6f}")
    
except Exception as e:
    print(f"   ✗ Error during inverse transform: {e}")
    y_test_unscaled = None
    y_pred_unscaled = None

# 8. Compare with Linear Regression
print("\n8. COMPARING WITH LINEAR REGRESSION...")
ml_metrics_path = os.path.join(MODELS, 'ml_models_metrics.json')
with open(ml_metrics_path, 'r') as f:
    ml_metrics = json.load(f)

print(f"   Linear Regression MAE: {ml_metrics['all_models']['Linear Regression']['mae']:.6f}")
print(f"   Linear Regression R²: {ml_metrics['all_models']['Linear Regression']['r2']:.6f}")

# 9. Sample predictions comparison
print("\n9. SAMPLE PREDICTIONS (First 10)...")
print(f"{'Index':<8} {'Actual (scaled)':<18} {'Pred (scaled)':<18} {'Actual (EUR/USD)':<18} {'Pred (EUR/USD)':<18}")
print("-" * 90)
for i in range(min(10, len(y_test))):
    if y_test_unscaled is not None:
        print(f"{i:<8} {y_test[i][0]:<18.6f} {y_pred_scaled[i][0]:<18.6f} {y_test_unscaled[i][0]:<18.4f} {y_pred_unscaled[i][0]:<18.4f}")
    else:
        print(f"{i:<8} {y_test[i][0]:<18.6f} {y_pred_scaled[i][0]:<18.6f} {'N/A':<18} {'N/A':<18}")

# 10. Diagnosis Summary
print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

if r2_scaled < 0:
    print("❌ PROBLEM: R² in scaled space is NEGATIVE!")
    print("   This means the model performs worse than predicting the mean.")
    
if y_test_unscaled is not None:
    if r2_unscaled < 0:
        print("❌ PROBLEM: R² in unscaled space is also NEGATIVE!")
        print("   The issue is not just a scaling problem.")
    else:
        print("✓ R² in unscaled space is positive!")
        print(f"  R² improved from {r2_scaled:.6f} to {r2_unscaled:.6f}")

# Check if metrics match what was reported
if abs(lstm_metrics['mae'] - mae_scaled) < 0.001:
    print(f"✓ Reported metrics match scaled predictions")
else:
    print(f"⚠ Reported MAE ({lstm_metrics['mae']:.6f}) doesn't match calculated ({mae_scaled:.6f})")

print("\nPOSSIBLE CAUSES:")
if r2_scaled < -1:
    print("  1. Model is making very poor predictions (not learning)")
    print("  2. Possible data leakage in training")
    print("  3. Model architecture may be inappropriate")
    print("  4. Training may not have converged")

print("\nRECOMMENDATIONS:")
print("  1. Check training loss curves - is the model learning?")
print("  2. Verify train/test split is chronological (no future data in training)")
print("  3. Try simpler baseline models (persistence model, moving average)")
print("  4. Consider if LSTM is appropriate for this problem")

print("\n" + "="*70)
