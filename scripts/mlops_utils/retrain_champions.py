"""
Production Model Retraining Script

This script retrains the top 3 models (champion, challenger, candidate) daily
using their exact feature configurations from MLflow. Designed for time series
forecasting where models need current data, not performance-based promotion.

Usage:
    python scripts/mlops_utils/retrain_champions.py
"""

import os
import sys
import json
import warnings
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import boto3
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
import tensorflow as tf

# Force CPU to avoid Metal issues on Mac
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

# Constants
EXPERIMENT_NAME = "EURUSD_Experiments"
TOP_N_MODELS = int(os.getenv('RETRAIN_TOP_N', 3))
MLFLOW_EC2_TAG = os.getenv('MLFLOW_EC2_TAG_NAME', 'MLflow-Server')


def get_mlflow_tracking_uri():
    """
    Discover MLflow EC2 public DNS dynamically.
    
    Returns:
        str: MLflow tracking URI (http://<public-dns>:5000)
    """
    # Check if running locally (use env var if set)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        print(f"Using MLflow URI from environment: {tracking_uri}")
        return tracking_uri
    
    # Running on AWS - discover MLflow server
    print(f"Discovering MLflow server with tag: Name={MLFLOW_EC2_TAG}")
    try:
        ec2 = boto3.client('ec2')
        response = ec2.describe_instances(
            Filters=[
                {'Name': 'tag:Name', 'Values': [MLFLOW_EC2_TAG]},
                {'Name': 'instance-state-name', 'Values': ['running']}
            ]
        )
        
        if not response['Reservations']:
            raise Exception(f"No running EC2 instance found with tag Name={MLFLOW_EC2_TAG}")
        
        public_dns = response['Reservations'][0]['Instances'][0]['PublicDnsName']
        mlflow_uri = f"http://{public_dns}:5000"
        print(f"Discovered MLflow server: {mlflow_uri}")
        return mlflow_uri
        
    except Exception as e:
        print(f"ERROR: Failed to discover MLflow server: {e}")
        print("Falling back to local MLflow (sqlite:///mlflow.db)")
        return "sqlite:///mlflow.db"


def get_champion_models(client):
    """
    Retrieve top N models from MLflow by alias.
    
    Args:
        client: MLflow tracking client
        
    Returns:
        list: Model metadata dicts with name, version, run_id, alias
    """
    print(f"\nRetrieving top {TOP_N_MODELS} models from MLflow...")
    
    aliases = ['champion', 'challenger', 'candidate'][:TOP_N_MODELS]
    models = []
    
    # Get all registered models
    registered_models = client.search_registered_models()
    
    for alias in aliases:
        print(f"\nLooking for @{alias}...")
        
        for rm in registered_models:
            model_name = rm.name
            
            try:
                # Get model version by alias
                model_version = client.get_model_version_by_alias(model_name, alias)
                
                models.append({
                    'name': model_name,
                    'version': model_version.version,
                    'run_id': model_version.run_id,
                    'alias': alias,
                    'model_type': model_name.replace('EURUSD_', '')
                })
                
                print(f"  Found: {model_name} v{model_version.version} (run: {model_version.run_id})")
                break  # Found this alias, move to next
                
            except Exception:
                continue  # This model doesn't have this alias
    
    if not models:
        raise Exception("No champion models found in MLflow. Run src/04_evaluate_select.py first.")
    
    print(f"\nFound {len(models)} models to retrain")
    return models


def download_feature_config(client, run_id):
    """
    Download feature_config.json from MLflow run.
    
    Args:
        client: MLflow tracking client
        run_id: MLflow run ID
        
    Returns:
        dict: Feature configuration with features, model_type, etc.
    """
    print(f"  Downloading feature_config.json from run {run_id}...")
    
    try:
        # Download artifact
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="feature_config.json"
        )
        
        with open(artifact_path, 'r') as f:
            config = json.load(f)
        
        print(f"    Features: {config.get('features', [])}")
        print(f"    Model type: {config.get('model_type', 'Unknown')}")
        
        return config
        
    except Exception as e:
        print(f"  WARNING: Could not download feature_config.json: {e}")
        return None


def eval_metrics(actual, pred):
    """Calculate evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    
    # Directional Accuracy
    actual_arr = np.array(actual)
    pred_arr = np.array(pred)
    start_sign = np.sign(actual_arr)
    pred_sign = np.sign(pred_arr)
    accuracy = np.mean(start_sign == pred_sign)
    
    return rmse, mae, accuracy


def retrain_linear_regression(feature_config, train_df, test_df, scaler_path):
    """
    Retrain Linear Regression model.
    
    Args:
        feature_config: Feature configuration dict
        train_df: Training dataframe
        test_df: Test dataframe
        scaler_path: Path to scaler artifact
        
    Returns:
        tuple: (model, predictions, metrics_dict)
    """
    print("\n  Training Linear Regression...")
    
    features = feature_config['features']
    target = feature_config.get('target', 'Target')
    
    # Filter features that exist
    features = [f for f in features if f in train_df.columns]
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    # Train
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict
    predictions = model.predict(X_test)
    rmse, mae, da = eval_metrics(y_test, predictions)
    
    print(f"    RMSE: {rmse:.5f}")
    print(f"    MAE: {mae:.5f}")
    print(f"    Directional Accuracy: {da:.3f}")
    
    # Log to MLflow
    with mlflow.start_run(run_name=f"LinearRegression_Retrain_{datetime.utcnow().strftime('%Y%m%d')}") as run:
        mlflow.set_tag("model_type", "LinearRegression")
        mlflow.set_tag("retrain", "production")
        mlflow.set_tag("retrain_date", datetime.utcnow().isoformat())
        
        mlflow.log_params(model.get_params())
        mlflow.log_param("features", features)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("directional_accuracy", da)
        
        mlflow.log_artifact(scaler_path, artifact_path="scaler")
        mlflow.log_dict(feature_config, "feature_config.json")
        
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(model, name="model", signature=signature)
        
        return model, predictions, {'rmse': rmse, 'mae': mae, 'da': da, 'run_id': run.info.run_id}


def retrain_arima(feature_config, train_df, test_df):
    """
    Retrain ARIMA model.
    
    Args:
        feature_config: Feature configuration dict
        train_df: Training dataframe
        test_df: Test dataframe
        
    Returns:
        tuple: (model, predictions, metrics_dict)
    """
    print("\n  Training ARIMA...")
    
    target = feature_config.get('target', 'Target')
    
    # ARIMA uses Return_Unscaled
    train_series = train_df['Return_Unscaled']
    test_series = map(float, test_df['Return_Unscaled'].values)
    
    # Fit Auto ARIMA
    model = auto_arima(
        train_series,
        seasonal=False,
        trend='c',
        trace=False,
        error_action='ignore',
        suppress_warnings=True
    )
    
    print(f"    Best Order: {model.order}")
    
    # Rolling forecast
    predictions = []
    for obs in test_series:
        model.update(obs)
        pred = model.predict(n_periods=1)[0]
        predictions.append(pred)
    
    predictions = np.array(predictions)
    y_test = test_df[target]
    rmse, mae, da = eval_metrics(y_test, predictions)
    
    print(f"    RMSE: {rmse:.5f}")
    print(f"    MAE: {mae:.5f}")
    print(f"    Directional Accuracy: {da:.3f}")
    
    # Log to MLflow
    with mlflow.start_run(run_name=f"ARIMA_Retrain_{datetime.utcnow().strftime('%Y%m%d')}") as run:
        mlflow.set_tag("model_type", "ARIMA")
        mlflow.set_tag("retrain", "production")
        mlflow.set_tag("retrain_date", datetime.utcnow().isoformat())
        
        mlflow.log_param("order", str(model.order))
        mlflow.log_param("seasonal_order", str(model.seasonal_order))
        mlflow.log_param("aic", model.aic())
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("directional_accuracy", da)
        
        mlflow.log_dict(feature_config, "feature_config.json")
        mlflow.sklearn.log_model(model, name="model")
        
        return model, predictions, {'rmse': rmse, 'mae': mae, 'da': da, 'run_id': run.info.run_id}


def retrain_lstm(feature_config, train_df, test_df, scaler_path):
    """
    Retrain LSTM model.
    
    Args:
        feature_config: Feature configuration dict
        train_df: Training dataframe
        test_df: Test dataframe
        scaler_path: Path to scaler artifact
        
    Returns:
        tuple: (model, predictions, metrics_dict)
    """
    print("\n  Training LSTM...")
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
    
    features = feature_config['features']
    target = feature_config.get('target', 'Target')
    time_steps = feature_config.get('time_steps', 60)
    
    # Filter features that exist
    features = [f for f in features if f in train_df.columns]
    n_features = len(features)
    
    print(f"    Features: {features}")
    print(f"    Time steps: {time_steps}")
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    # Create sequences
    def create_sequences(data, target, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps + 1):
            X.append(data[i:(i + time_steps)])
            y.append(target[i + time_steps - 1])
        return np.array(X), np.array(y)
    
    X_train_seq, y_train_seq = create_sequences(X_train.values, y_train.values, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test.values, y_test.values, time_steps)
    
    print(f"    Input shape: {X_train_seq.shape}")
    
    # Build model
    epochs = 20
    batch_size = 32
    
    model = Sequential()
    model.add(Input(shape=(time_steps, n_features)))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train
    model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Predict
    predictions = model.predict(X_test_seq, verbose=0)
    predictions = predictions.flatten()
    
    rmse, mae, da = eval_metrics(y_test_seq, predictions)
    
    print(f"    RMSE: {rmse:.5f}")
    print(f"    MAE: {mae:.5f}")
    print(f"    Directional Accuracy: {da:.3f}")
    
    # Log to MLflow
    with mlflow.start_run(run_name=f"LSTM_Retrain_{datetime.utcnow().strftime('%Y%m%d')}") as run:
        mlflow.set_tag("model_type", "LSTM")
        mlflow.set_tag("retrain", "production")
        mlflow.set_tag("retrain_date", datetime.utcnow().isoformat())
        
        mlflow.log_param("input_shape", f"[{X_train_seq.shape[0]}, {time_steps}, {n_features}]")
        mlflow.log_param("time_steps", time_steps)
        mlflow.log_param("n_features", n_features)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("directional_accuracy", da)
        
        mlflow.log_artifact(scaler_path, artifact_path="scaler")
        mlflow.log_dict(feature_config, "feature_config.json")
        
        signature = infer_signature(X_train_seq, predictions)
        mlflow.tensorflow.log_model(model, name="model", signature=signature)
        
        return model, predictions, {'rmse': rmse, 'mae': mae, 'da': da, 'run_id': run.info.run_id}


def update_model_alias(client, model_name, new_version, alias):
    """
    Update model alias to new version.
    
    Args:
        client: MLflow tracking client
        model_name: Registered model name
        new_version: New model version to assign alias
        alias: Alias to update (champion, challenger, candidate)
    """
    print(f"\n  Updating @{alias} alias for {model_name} to version {new_version}")
    
    try:
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=new_version
        )
        print(f"    ✅ Updated successfully")
    except Exception as e:
        print(f"    ❌ Failed to update alias: {e}")


def main():
    """Main retraining workflow."""
    print("="*80)
    print("PRODUCTION MODEL RETRAINING")
    print(f"Started at: {datetime.utcnow().isoformat()} UTC")
    print("="*80)
    
    # 1. Setup MLflow
    tracking_uri = get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = mlflow.tracking.MlflowClient()
    
    # 2. Get champion models
    try:
        champion_models = get_champion_models(client)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    
    # 3. Load latest data
    print("\n" + "="*80)
    print("LOADING LATEST DATA")
    print("="*80)
    
    try:
        from utils.data_manager import DataManager
        dm = DataManager(data_type='processed')
        train_df, test_df, scaler = dm.load_processed()
        
        if train_df is None or test_df is None:
            raise Exception("Failed to load processed data. Run src/02_preprocess.py first.")
        
        scaler_path = dm.get_local_path('scaler.pkl')
        
        print(f"✅ Loaded data successfully")
        print(f"   Train shape: {train_df.shape}")
        print(f"   Test shape: {test_df.shape}")
        
    except Exception as e:
        print(f"\nERROR: Failed to load data: {e}")
        sys.exit(1)
    
    # 4. Retrain each model
    print("\n" + "="*80)
    print("RETRAINING MODELS")
    print("="*80)
    
    retrained_models = []
    
    for model_info in champion_models:
        print(f"\n{'='*80}")
        print(f"RETRAINING: {model_info['name']} (@{model_info['alias']})")
        print(f"{'='*80}")
        
        # Download feature config
        feature_config = download_feature_config(client, model_info['run_id'])
        
        if not feature_config:
            print(f"  ⚠️  Skipping {model_info['name']} - no feature config found")
            continue
        
        # Retrain based on model type
        try:
            model_type = model_info['model_type']
            
            if 'Linear' in model_type:
                model, preds, metrics = retrain_linear_regression(
                    feature_config, train_df, test_df, scaler_path
                )
            elif 'ARIMA' in model_type:
                model, preds, metrics = retrain_arima(
                    feature_config, train_df, test_df
                )
            elif 'LSTM' in model_type:
                model, preds, metrics = retrain_lstm(
                    feature_config, train_df, test_df, scaler_path
                )
            else:
                print(f"  ⚠️  Unknown model type: {model_type}")
                continue
            
            # Register new model version
            model_uri = f"runs:/{metrics['run_id']}/model"
            reg_model = mlflow.register_model(model_uri, model_info['name'])
            
            retrained_models.append({
                'name': model_info['name'],
                'alias': model_info['alias'],
                'old_version': model_info['version'],
                'new_version': reg_model.version,
                'metrics': metrics
            })
            
            print(f"  ✅ Registered as version {reg_model.version}")
            
        except Exception as e:
            print(f"  ❌ Failed to retrain {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 5. Update aliases (always, for time series freshness)
    print("\n" + "="*80)
    print("UPDATING MODEL ALIASES")
    print("="*80)
    
    for model in retrained_models:
        update_model_alias(
            client,
            model['name'],
            model['new_version'],
            model['alias']
        )
    
    # 6. Summary
    print("\n" + "="*80)
    print("RETRAINING SUMMARY")
    print("="*80)
    
    for model in retrained_models:
        print(f"\n{model['name']}:")
        print(f"  Alias: @{model['alias']}")
        print(f"  Version: {model['old_version']} → {model['new_version']}")
        print(f"  RMSE: {model['metrics']['rmse']:.5f}")
        print(f"  MAE: {model['metrics']['mae']:.5f}")
        print(f"  Directional Accuracy: {model['metrics']['da']:.3f}")
    
    print("\n" + "="*80)
    print(f"COMPLETED at: {datetime.utcnow().isoformat()} UTC")
    print("="*80)


if __name__ == "__main__":
    main()
