#!/usr/bin/env python3
"""
Model Comparison Script
================================================
Compares different versions of models (e.g., Best vs Latest) to analyze trade-offs.

Usage:
    python tests/compare_models.py

Supported Modes:
    - 'best_vs_latest': Compares the Champion (lowest RMSE) vs Candidate (most recent)
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

# Re-use functions from check_overfitting if possible, but for standalone safety we copy core logic
# or we could import them if check_overfitting was a module. 
# For simplicity, I will copy the core evaluation logic here.

load_dotenv()
warnings.filterwarnings("ignore")

# Constants
EXPERIMENT_NAME = "EURUSD_Experiments"

# --- Shared Logic (Same as check_overfitting.py) ---

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    actual_arr = np.array(actual)
    pred_arr = np.array(pred)
    start_sign = np.sign(actual_arr)
    pred_sign = np.sign(pred_arr)
    da = np.mean(start_sign == pred_sign)
    return rmse, mae, r2, da

def print_header(text):
    print("\n" + "="*70)
    print(text)
    print("="*70)

def print_subheader(text):
    print(f"\n{text}")
    print("-" * 70)

def diagnose_overfitting(train_metrics, test_metrics):
    train_rmse, train_mae, train_r2, _ = train_metrics
    test_rmse, test_mae, test_r2, _ = test_metrics
    
    if train_rmse is None or test_rmse is None:
        return "UNKNOWN"
        
    r2_diff = train_r2 - test_r2
    mae_ratio = test_mae / train_mae if train_mae > 0 else float('inf')
    
    if r2_diff < 0.05 and 0.9 < mae_ratio < 1.1:
        return "GOOD"
    elif r2_diff < 0.1:
        return "MILD"
    else:
        return "SEVERE"

# --- Model Check Functions ---

def analyze_linear_regression(run_id, train_df, test_df):
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    feature_config_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="feature_config.json")
    import json
    with open(feature_config_path, 'r') as f:
        feature_config = json.load(f)
    features = feature_config['features']
    target = feature_config['target']
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_metrics = eval_metrics(y_train, y_train_pred)
    test_metrics = eval_metrics(y_test, y_test_pred)
    
    return train_metrics, test_metrics, features

def analyze_arima(run_id, train_df, test_df):
    model_uri = f"runs:/{run_id}/model"
    arima_model = mlflow.sklearn.load_model(model_uri)
    target = 'Target'
    col_to_use = 'Return_Unscaled'
    
    train_metrics = (None, None, None, None)
    # Train predict
    try:
        y_train_pred = arima_model.predict_in_sample()
        if len(y_train_pred) == len(train_df):
            train_metrics = eval_metrics(train_df[target], y_train_pred)
    except:
        pass
        
    # Test predict
    test_predictions = []
    arima_test = arima_model
    for obs in test_df[col_to_use].values:
        arima_test.update(obs)
        pred = arima_test.predict(n_periods=1)[0]
        test_predictions.append(pred)
    
    test_metrics = eval_metrics(test_df[target], test_predictions)
    return train_metrics, test_metrics, ["ARIMA_Order"]

def analyze_lstm(run_id, train_df, test_df):
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.tensorflow.load_model(model_uri)
    feature_config_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="feature_config.json")
    import json
    with open(feature_config_path, 'r') as f:
        feature_config = json.load(f)
    features = feature_config['features']
    target = feature_config['target']
    time_steps = feature_config['time_steps']
    
    def create_sequences(data, target_data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps + 1):
            X.append(data[i:(i + time_steps)])
            y.append(target_data[i + time_steps - 1])
        return np.array(X), np.array(y)
        
    X_train_seq, y_train_seq = create_sequences(train_df[features].values, train_df[target].values, time_steps)
    X_test_seq, y_test_seq = create_sequences(test_df[features].values, test_df[target].values, time_steps)
    
    y_train_pred = model.predict(X_train_seq, verbose=0).flatten()
    y_test_pred = model.predict(X_test_seq, verbose=0).flatten()
    
    train_metrics = eval_metrics(y_train_seq, y_train_pred)
    test_metrics = eval_metrics(y_test_seq, y_test_pred)
    
    return train_metrics, test_metrics, features

# --- Main Logic ---

def main():
    print_header("MODEL COMPARISON TOOL")
    
    # 1. Setup
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        print(f"📡 Using Remote MLflow: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
    else:
        print("💾 Using Local MLflow")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
    # 2. Add Project to Path to load DataManager
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    from utils.data_manager import DataManager
    
    dm = DataManager(data_type='processed')
    train_df, test_df, scaler = dm.load_processed()
    
    if train_df is None:
        print("❌ Could not load data.")
        return

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    
    if not experiment:
        print("Experiment not found.")
        return

    # 3. Fetch Runs
    print("\n🔍 Fetching Runs...")
    runs_best = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.rmse ASC"])
    runs_latest = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["attribute.start_time DESC"])
    
    models_to_compare = {} # { 'LinearRegression': {'Best': run, 'Latest': run} }
    
    def register_run(run_list, label):
        for run in run_list:
            m_type = run.data.params.get('model_type') or run.data.tags.get('model_type')
            if m_type:
                if m_type not in models_to_compare:
                    models_to_compare[m_type] = {}
                if label not in models_to_compare[m_type]:
                    models_to_compare[m_type][label] = run

    register_run(runs_best, "Best")
    register_run(runs_latest, "Latest")
    
    # 4. Compare
    for m_type, variants in models_to_compare.items():
        print_header(f"COMPARING: {m_type}")
        
        # Table Header
        print(f"{'Variant':<10} {'Run ID':<10} {'Test RMSE':<12} {'Test R²':<12} {'Overfitting':<12} {'Features'}")
        print("-" * 100)
        
        for variant_name in ['Best', 'Latest']:
            if variant_name not in variants:
                continue
                
            run = variants[variant_name]
            run_id = run.info.run_id
            
            # Analyze
            try:
                if m_type == 'LinearRegression':
                    tr_m, te_m, feats = analyze_linear_regression(run_id, train_df, test_df)
                elif m_type == 'ARIMA':
                    tr_m, te_m, feats = analyze_arima(run_id, train_df, test_df)
                elif m_type == 'LSTM':
                    tr_m, te_m, feats = analyze_lstm(run_id, train_df, test_df)
                else:
                    continue
                
                status = diagnose_overfitting(tr_m, te_m)
                
                # Format output
                rmse_str = f"{te_m[0]:.6f}" if te_m[0] else "N/A"
                r2_str = f"{te_m[2]:.4f}" if te_m[2] else "N/A"
                feat_count = len(feats)
                
                print(f"{variant_name:<10} {run_id[:8]:<10} {rmse_str:<12} {r2_str:<12} {status:<12} {feat_count} features")
                
            except Exception as e:
                print(f"{variant_name:<10} {run_id[:8]:<10} ERROR: {str(e)}")

if __name__ == "__main__":
    main()
