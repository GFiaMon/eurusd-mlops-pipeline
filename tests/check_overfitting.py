#!/usr/bin/env python3
"""
Overfitting Analysis Script (MLflow-Compatible)
================================================
Checks if models are overfitting by comparing train vs test performance.
Compatible with the current MLflow-based pipeline where models are stored in S3.

This script:
- Loads models from MLflow Model Registry (local or remote)
- Uses the same data split as the training pipeline (80/20)
- Supports Linear Regression, ARIMA, and LSTM models
- Provides detailed overfitting diagnostics
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv

load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.data_manager import DataManager

# Constants
EXPERIMENT_NAME = "EURUSD_Experiments"

def check_environment():
    """Verify that necessary environment variables are set for remote access"""
    print_header("ENVIRONMENT CHECK")
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    issues = []
    
    if not tracking_uri:
        issues.append("❌ MLFLOW_TRACKING_URI is not set. Script will default to local, but you likely want REMOTE.")
    elif "localhost" in tracking_uri or "127.0.0.1" in tracking_uri:
        print(f"⚠️  MLFLOW_TRACKING_URI points to localhost ({tracking_uri}). Ensure your SSH tunnel is active if accessing EC2.")
    else:
        print(f"✅ MLflow Tracking URI set: {tracking_uri}")
        
    # Removed strict AWS Env Var check since boto3 finds ~/.aws/credentials automatically.
        
    if issues:
        print("\nPossible Configuration Issues:")
        for issue in issues:
            print(issue)
        print("\nContinuing, but errors may occur...")
    else:
        print("\nEnvironment looks good for remote access.")

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(text)
    print("="*70)

def print_subheader(text):
    """Print a formatted subheader"""
    print(f"\n{text}")
    print("-" * 70)

def eval_metrics(actual, pred):
    """Calculate evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    
    # Directional Accuracy
    actual_arr = np.array(actual)
    pred_arr = np.array(pred)
    start_sign = np.sign(actual_arr)
    pred_sign = np.sign(pred_arr)
    da = np.mean(start_sign == pred_sign)
    
    return rmse, mae, r2, da

def diagnose_overfitting(train_metrics, test_metrics, model_type):
    """Diagnose overfitting based on train/test metrics"""
    train_rmse, train_mae, train_r2, train_da = train_metrics
    test_rmse, test_mae, test_r2, test_da = test_metrics
    
    print_subheader("OVERFITTING DIAGNOSIS")
    
    if train_rmse is None or test_rmse is None:
        print("\n⚠️  Cannot diagnose overfitting: Missing metrics (likely shape mismatch).")
        return "UNKNOWN"
    
    r2_diff = train_r2 - test_r2
    mae_ratio = test_mae / train_mae if train_mae > 0 else float('inf')
    rmse_ratio = test_rmse / train_rmse if train_rmse > 0 else float('inf')
    
    print(f"\nR² Difference (train - test): {r2_diff:.6f}")
    print(f"MAE Ratio (test / train): {mae_ratio:.4f}")
    print(f"RMSE Ratio (test / train): {rmse_ratio:.4f}")
    
    # Diagnosis logic
    if r2_diff < 0.05 and 0.9 < mae_ratio < 1.1:
        print("\n✅ GOOD GENERALIZATION - No significant overfitting detected!")
        print("   - R² difference is small (< 0.05)")
        print("   - Test and train errors are similar")
        print("   - Model generalizes well to unseen data")
        return "GOOD"
    elif r2_diff < 0.1:
        print("\n⚠️  MILD OVERFITTING - Some overfitting detected")
        print("   - R² difference is moderate (0.05-0.10)")
        print("   - Model may benefit from regularization")
        return "MILD"
    else:
        print("\n❌ SIGNIFICANT OVERFITTING - Model is overfitting!")
        print("   - R² difference is large (> 0.10)")
        print("   - Model memorizing training data")
        print("   - Regularization strongly recommended")
        return "SEVERE"

def check_linear_regression_overfitting(run_id, train_df, test_df, scaler):
    """Check overfitting for Linear Regression model"""
    print_header("LINEAR REGRESSION OVERFITTING ANALYSIS")
    
    # Load model from MLflow
    print("\n1. LOADING MODEL FROM MLFLOW...")
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Load feature config
    print("   Downloading feature_config.json...")
    feature_config_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, 
        artifact_path="feature_config.json"
    )
    import json
    with open(feature_config_path, 'r') as f:
        feature_config = json.load(f)
    
    features = feature_config['features']
    target = feature_config['target']
    
    print(f"   Model loaded successfully")
    print(f"   Features: {features}")
    
    # Prepare data
    print("\n2. PREPARING DATA...")
    try:
        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]
        
        print(f"   Train samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        # Make predictions
        print("\n3. MAKING PREDICTIONS...")
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        print("\n4. CALCULATING METRICS...")
        train_metrics = eval_metrics(y_train, y_train_pred)
        test_metrics = eval_metrics(y_test, y_test_pred)
        
        train_rmse, train_mae, train_r2, train_da = train_metrics
        test_rmse, test_mae, test_r2, test_da = test_metrics
        
        # Display results
        print_subheader("PERFORMANCE COMPARISON")
        print(f"\n{'Metric':<20} {'Train':<15} {'Test':<15} {'Difference':<15}")
        print("-" * 70)
        print(f"{'RMSE':<20} {train_rmse:<15.6f} {test_rmse:<15.6f} {abs(train_rmse - test_rmse):<15.6f}")
        print(f"{'MAE':<20} {train_mae:<15.6f} {test_mae:<15.6f} {abs(train_mae - test_mae):<15.6f}")
        print(f"{'R²':<20} {train_r2:<15.6f} {test_r2:<15.6f} {abs(train_r2 - test_r2):<15.6f}")
        print(f"{'Dir. Accuracy':<20} {train_da:<15.6f} {test_da:<15.6f} {abs(train_da - test_da):<15.6f}")
        
        # Overfitting diagnosis
        status = diagnose_overfitting(train_metrics, test_metrics, "LinearRegression")
        
        # Residual analysis
        print_subheader("RESIDUAL ANALYSIS")
        train_residuals = y_train - y_train_pred
        test_residuals = y_test - y_test_pred
        
        print(f"\nTrain residuals mean: {train_residuals.mean():.6f} (should be ~0)")
        print(f"Test residuals mean: {test_residuals.mean():.6f} (should be ~0)")
        print(f"Train residuals std: {train_residuals.std():.6f}")
        print(f"Test residuals std: {test_residuals.std():.6f}")
        
        if abs(test_residuals.mean()) > 0.001:
            print("\n⚠️  WARNING: Test residuals have non-zero mean")
            print("   This suggests systematic bias in predictions")
        
        # Feature importance
        print_subheader("FEATURE IMPORTANCE (TOP 10)")
        feature_importance = pd.DataFrame({
            'feature': features,
            'coefficient': model.coef_
        })
        feature_importance['abs_coefficient'] = abs(feature_importance['coefficient'])
        feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
        
        print(f"\n{'Feature':<20} {'Coefficient':<15} {'Impact':<10}")
        print("-" * 50)
        for idx, row in feature_importance.head(10).iterrows():
            impact = "↑ Positive" if row['coefficient'] > 0 else "↓ Negative"
            print(f"{row['feature']:<20} {row['coefficient']:<15.6f} {impact:<10}")
        
        return status
        
    except KeyError as e:
        print(f"\n❌ ERROR: Missing features in processed data.")
        print(f"   Model expects: {features}")
        print(f"   Data contains: {list(train_df.columns)}")
        print(f"   Missing: {e}")
        print("\n   Try running 'python src/02_preprocess.py' to regenerate data.")
        return "ERROR"

def check_arima_overfitting(run_id, train_df, test_df):
    """Check overfitting for ARIMA model"""
    print_header("ARIMA OVERFITTING ANALYSIS")
    
    # Load model from MLflow
    print("\n1. LOADING MODEL FROM MLFLOW...")
    model_uri = f"runs:/{run_id}/model"
    arima_model = mlflow.sklearn.load_model(model_uri)
    
    print(f"   Model loaded successfully")
    print(f"   Order: {arima_model.order}")
    
    # Prepare data
    print("\n2. PREPARING DATA...")
    train_series = train_df['Return']  # Note: 02_preprocess now produces unscaled Return as 'Return' implicitly or explicit?
    # Wait, src/03_train_models.py uses 'Return_Unscaled'. 
    # Let's check columns. preprocess.py says: features generated: ['Return', ...]
    # But it also does `train_df['Return_Unscaled'] = train_df['Return']` BEFORE scaling in preprocess.py?
    # Wait, in preprocess.py, `train_df['Return_Unscaled']` is NOT in the `feature_cols` list so it is NOT scaled.
    # But is it saved?
    # dm_processed.save_processed(train_df...) saves the dataframe.
    # Columns saved are whatever is in train_df.
    # BUT `scaler.fit_transform` only modifies `feature_cols`.
    # Does train_df still contain 'Return_Unscaled'?
    
    # Let's try to use 'Return' but inverse transform if needed.
    # Actually, let's assume 'Return' IS scaled in the file.
    # ARIMA needs unscaled returns usually? Or if it was trained on unscaled.
    # The training script used 'Return_Unscaled'.
    # If 'Return_Unscaled' is not in columns, we might need to rely on 'Return' (scaled) if that's what we have.
    # However, let's verify if 'Return_Unscaled' is in train_df.
    
    target = 'Target'
    
    col_to_use = 'Return_Unscaled'
    if col_to_use not in train_df.columns:
        print(f"⚠️  '{col_to_use}' not found. Using 'Return' (Note: might be scaled).")
        col_to_use = 'Return'
        
    train_series = train_df[col_to_use]
    test_series = test_df[col_to_use]
    
    print(f"   Train samples: {len(train_series)}")
    print(f"   Test samples: {len(test_series)}")
    
    # Make predictions on train (in-sample)
    print("\n3. MAKING PREDICTIONS...")
    print("   Train predictions (in-sample)...")
    
    train_metrics = (0, 0, 0, 0)
    try:
        y_train_pred = arima_model.predict_in_sample()
        
        # Check alignment
        if len(y_train_pred) != len(train_series):
            print(f"   ⚠️  Shape Mismatch: Model fit on {len(y_train_pred)} samples, but current train set has {len(train_series)}.")
            print("   Skipping Train metrics calculation (cannot compare).")
            train_metrics = (None, None, None, None)
        else:
            y_train = train_df[target].values
            train_metrics = eval_metrics(y_train, y_train_pred)
            
    except Exception as e:
        print(f"   ⚠️  Could not calculate in-sample predictions: {e}")
        train_metrics = (None, None, None, None) # placeholders
    
    # Rolling forecast for test
    print("   Test predictions (rolling forecast)...")
    test_predictions = []
    arima_test = arima_model
    
    for obs in test_series.values:
        arima_test.update(obs)
        pred = arima_test.predict(n_periods=1)[0]
        test_predictions.append(pred)
    
    test_predictions = np.array(test_predictions)
    
    y_test = test_df[target].values
    
    # Calculate metrics
    print("\n4. CALCULATING METRICS...")
    # train_metrics is already computed or set to 0
    test_metrics = eval_metrics(y_test, test_predictions)
    
    train_rmse, train_mae, train_r2, train_da = train_metrics
    test_rmse, test_mae, test_r2, test_da = test_metrics
    
    # Display results
    print_subheader("PERFORMANCE COMPARISON")
    print(f"\n{'Metric':<20} {'Train':<15} {'Test':<15} {'Difference':<15}")
    print("-" * 70)
    
    def fmt(val): return f"{val:.6f}" if val is not None else "N/A"
    def diff(v1, v2): return f"{abs(v1 - v2):.6f}" if (v1 is not None and v2 is not None) else "N/A"

    print(f"{'RMSE':<20} {fmt(train_rmse):<15} {fmt(test_rmse):<15} {diff(train_rmse, test_rmse):<15}")
    print(f"{'MAE':<20} {fmt(train_mae):<15} {fmt(test_mae):<15} {diff(train_mae, test_mae):<15}")
    print(f"{'R²':<20} {fmt(train_r2):<15} {fmt(test_r2):<15} {diff(train_r2, test_r2):<15}")
    print(f"{'Dir. Accuracy':<20} {fmt(train_da):<15} {fmt(test_da):<15} {diff(train_da, test_da):<15}")
    
    # Overfitting diagnosis
    status = diagnose_overfitting(train_metrics, test_metrics, "ARIMA")
    
    return status

def check_lstm_overfitting(run_id, train_df, test_df, scaler):
    """Check overfitting for LSTM model"""
    print_header("LSTM OVERFITTING ANALYSIS")
    
    # Load model from MLflow
    print("\n1. LOADING MODEL FROM MLFLOW...")
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.tensorflow.load_model(model_uri)
    
    # Load feature config
    print("   Downloading feature_config.json...")
    feature_config_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, 
        artifact_path="feature_config.json"
    )
    import json
    with open(feature_config_path, 'r') as f:
        feature_config = json.load(f)
    
    features = feature_config['features']
    target = feature_config['target']
    time_steps = feature_config['time_steps']
    
    print(f"   Model loaded successfully")
    print(f"   Features: {features}")
    print(f"   Time steps: {time_steps}")
    
    # Prepare data
    print("\n2. PREPARING DATA...")
    
    def create_sequences(data, target_data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps + 1):
            X.append(data[i:(i + time_steps)])
            y.append(target_data[i + time_steps - 1])
        return np.array(X), np.array(y)
    
    try:
        X_train_lstm = train_df[features].values
        y_train_lstm = train_df[target].values
        X_test_lstm = test_df[features].values
        y_test_lstm = test_df[target].values
        
        X_train_seq, y_train_seq = create_sequences(X_train_lstm, y_train_lstm, time_steps)
        X_test_seq, y_test_seq = create_sequences(X_test_lstm, y_test_lstm, time_steps)
        
        print(f"   Train sequences: {len(X_train_seq)}")
        print(f"   Test sequences: {len(X_test_seq)}")
        
        # Make predictions
        print("\n3. MAKING PREDICTIONS...")
        y_train_pred = model.predict(X_train_seq, verbose=0).flatten()
        y_test_pred = model.predict(X_test_seq, verbose=0).flatten()
        
        # Calculate metrics
        print("\n4. CALCULATING METRICS...")
        train_metrics = eval_metrics(y_train_seq, y_train_pred)
        test_metrics = eval_metrics(y_test_seq, y_test_pred)
        
        train_rmse, train_mae, train_r2, train_da = train_metrics
        test_rmse, test_mae, test_r2, test_da = test_metrics
        
        # Display results
        print_subheader("PERFORMANCE COMPARISON")
        print(f"\n{'Metric':<20} {'Train':<15} {'Test':<15} {'Difference':<15}")
        print("-" * 70)
        print(f"{'RMSE':<20} {train_rmse:<15.6f} {test_rmse:<15.6f} {abs(train_rmse - test_rmse):<15.6f}")
        print(f"{'MAE':<20} {train_mae:<15.6f} {test_mae:<15.6f} {abs(train_mae - test_mae):<15.6f}")
        print(f"{'R²':<20} {train_r2:<15.6f} {test_r2:<15.6f} {abs(train_r2 - test_r2):<15.6f}")
        print(f"{'Dir. Accuracy':<20} {train_da:<15.6f} {test_da:<15.6f} {abs(train_da - test_da):<15.6f}")
        
        # Overfitting diagnosis
        status = diagnose_overfitting(train_metrics, test_metrics, "LSTM")
        
        # Residual analysis
        print_subheader("RESIDUAL ANALYSIS")
        train_residuals = y_train_seq - y_train_pred
        test_residuals = y_test_seq - y_test_pred
        
        print(f"\nTrain residuals mean: {train_residuals.mean():.6f} (should be ~0)")
        print(f"Test residuals mean: {test_residuals.mean():.6f} (should be ~0)")
        print(f"Train residuals std: {train_residuals.std():.6f}")
        print(f"Test residuals std: {test_residuals.std():.6f}")
        
        if abs(test_residuals.mean()) > 0.001:
            print("\n⚠️  WARNING: Test residuals have non-zero mean")
            print("   This suggests systematic bias in predictions")
        
        return status
        
    except KeyError as e:
        print(f"\n❌ ERROR: Missing features in processed data.")
        print(f"   Model expects: {features}")
        print(f"   Missing: {e}")
        return "ERROR"

def main():
    """Main function to check overfitting for all models"""
    print_header("MLFLOW-BASED OVERFITTING ANALYSIS")
    print("Checking overfitting for models stored in MLflow Model Registry")
    
    # Check Env
    check_environment()
    
    # Setup MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        print(f"\n📡 Using Remote MLflow: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
    else:
        print("\n⚠️  MLFLOW_TRACKING_URI not set. Falling back to local (sqlite:///mlflow.db).")
        print("    If you want to check AWS models, export MLFLOW_TRACKING_URI='http://<EC2_IP>:5000'")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Load data
    print("\n📥 LOADING DATA VIA DATAMANAGER...")
    dm = DataManager(data_type='processed')
    train_df, test_df, scaler = dm.load_processed()
    
    if train_df is None or test_df is None:
        print("❌ Error: Could not load processed data. Run src/02_preprocess.py first.")
        return
    
    print(f"   Train samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")
    
    # Get experiment
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    
    if experiment is None:
        print(f"❌ Error: Experiment '{EXPERIMENT_NAME}' not found. Run training first.")
        return
    
    # Find best runs per model type
    print("\n🔍 QUERYING MLFLOW FOR BEST MODELS...")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"]
    )
    
    best_runs = {}
    for run in runs:
        model_type = run.data.params.get('model_type') or run.data.tags.get('model_type')
        if model_type and model_type not in best_runs:
            best_runs[model_type] = run
    
    print(f"   Found {len(best_runs)} model types: {list(best_runs.keys())}")
    
    # Check overfitting for each model
    results = {}
    
    if 'LinearRegression' in best_runs:
        run = best_runs['LinearRegression']
        print(f"\n📊 Analyzing Linear Regression (Run ID: {run.info.run_id[:8]}...)")
        results['LinearRegression'] = check_linear_regression_overfitting(
            run.info.run_id, train_df, test_df, scaler
        )
    
    if 'ARIMA' in best_runs:
        run = best_runs['ARIMA']
        print(f"\n📊 Analyzing ARIMA (Run ID: {run.info.run_id[:8]}...)")
        results['ARIMA'] = check_arima_overfitting(
            run.info.run_id, train_df, test_df
        )
    
    if 'LSTM' in best_runs:
        run = best_runs['LSTM']
        print(f"\n📊 Analyzing LSTM (Run ID: {run.info.run_id[:8]}...)")
        results['LSTM'] = check_lstm_overfitting(
            run.info.run_id, train_df, test_df, scaler
        )
            
    # Summary
    print_header("SUMMARY")
    print("\nOverfitting Status by Model:")
    for model_type, status in results.items():
        emoji = "✅" if status == "GOOD" else "⚠️" if status == "MILD" else "❌"
        print(f"  {emoji} {model_type}: {status}")
    
    # Recommendations
    print_subheader("RECOMMENDATIONS")
    
    if all(s == "GOOD" for s in results.values()):
        print("\n✅ All models are performing well!")
        print("   - Continue monitoring performance on new data")
        print("   - Consider ensemble methods for further improvement")
    elif any(s == "SEVERE" for s in results.values()):
        print("\n❌ Some models show significant overfitting:")
        print("   1. Implement regularization (Ridge/Lasso for Linear Regression)")
        print("   2. Use dropout/early stopping for LSTM")
        print("   3. Perform feature selection to reduce dimensionality")
        print("   4. Use cross-validation to tune hyperparameters")
    else:
        print("\n⚠️  Some models show mild overfitting:")
        print("   1. Monitor performance on new data")
        print("   2. Consider adding regularization")
        print("   3. Validate with cross-validation")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
