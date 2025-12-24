import pandas as pd
import numpy as np
import os
import warnings
from dotenv import load_dotenv

load_dotenv()

# Suppress annoying "pkg_resources is deprecated" warning from mlflow dependencies
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout

# Force CPU to avoid Metal issues on Mac
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

# Constants
PROCESSED_DATA_DIR = os.path.join("data", "processed")
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "test.csv")
SCALER_PATH = os.path.join(PROCESSED_DATA_DIR, "scaler.pkl")
EXPERIMENT_NAME = "EURUSD_Experiments"

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    
    # Directional Accuracy: Fraction of times the sign of prediction matches sign of actual
    # Note: If actual is 0, sign is 0. If pred is 0, sign is 0.
    # We compare np.sign(actual) == np.sign(pred).
    # Use .values (or np.array) to avoid index mismatch errors if inputs are pandas Series
    actual_arr = np.array(actual)
    pred_arr = np.array(pred)
    
    start_sign = np.sign(actual_arr)
    pred_sign = np.sign(pred_arr)
    accuracy = np.mean(start_sign == pred_sign)
    
    return rmse, mae, accuracy

def train_models():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_DATA_PATH, index_col=0, parse_dates=True)
    test_df = pd.read_csv(TEST_DATA_PATH, index_col=0, parse_dates=True)
    
    # Updated Feature List
    feature_cols = ['Return', 'MA_5', 'MA_10', 'MA_20', 'MA_50', 'Return_5d', 'Return_20d', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5']
    # Add OHLC columns if they exist in the dataframe
    potential_cols = ['Open', 'High', 'Low', 'Close']
    for col in potential_cols:
        if col in train_df.columns:
            feature_cols.append(col)
            
    print(f"Features used for training: {feature_cols}")
    target_col = 'Target'
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # Setup MLflow
    # Note: connect to local MLflow (./mlruns) by default.
    # To use a remote server, set MLFLOW_TRACKING_URI env var or call mlflow.set_tracking_uri("http://...")
    # Connect to a local SQLite DB to enable the Model Registry
# Check if running on AWS (simple check)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        print(f"Using Remote MLflow Tracking URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
    else:
        print("Using Local MLflow (sqlite:///mlflow.db)")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Common Tags
    tags = {
        "developer": "User",
        "project": "EURUSD_Capstone",
        "data_version": "v1"
    }
    
    # --- Model A: Linear Regression ---
    with mlflow.start_run(run_name="Linear_Regression") as run:
        mlflow.set_tags(tags)
        mlflow.set_tag("model_type", "LinearRegression")
        
        print("Training Linear Regression...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        predictions = lr.predict(X_test)
        rmse, mae, da = eval_metrics(y_test, predictions)
        
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  Directional Accuracy: {da}")
        
        # Log Params
        mlflow.log_params(lr.get_params())
        mlflow.log_param("features", feature_cols)
        
        # Log Metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("directional_accuracy", da)
        
        # Log Scaler as Artifact
        mlflow.log_artifact(SCALER_PATH, artifact_path="scaler")
        
        # Log Feature Config
        feature_config = {
            "features": feature_cols,
            "target": target_col,
            "model_type": "LinearRegression"
        }
        mlflow.log_dict(feature_config, "feature_config.json")
        
        # Infer and log signature
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(lr, artifact_path="model", signature=signature)

    # --- Model B: ARIMA ---
    with mlflow.start_run(run_name="ARIMA") as run:
        mlflow.set_tags(tags)
        mlflow.set_tag("model_type", "ARIMA")
        
        print("Training ARIMA...")
        # For ARIMA, we use the 'Return_Unscaled' series from the training set to forecast.
        
        # Fit Auto ARIMA on 'Return_Unscaled'
        # Note regarding DA=0: If ARIMA(0,0,0) with no intercept is selected (Random Walk), it predicts 0.0 for everything.
        # This leads to Directional Accuracy of 0.0 because actuals are never exactly 0.
        # To fix this and improve RMSE, we use a Rolling Forecast (Walk-Forward Validation).
        # We model R_t. At step t (in test), we have history R_0...R_{t-1}. We predict R_t.
        # Our 'y_test' is Target_t = R_{t+1}.
        # Wait, proper alignment:
        # Train ends at T-1.
        # Test starts at T.
        # Test row 0 has 'Return_Unscaled' = R_T. And 'Target' = R_{T+1}.
        # To predict R_{T+1}, we need history up to R_T.
        # So for each step in Test:
        #   1. Update model with observed Return (R_T).
        #   2. Predict 1 step ahead -> R_{T+1}.
        #   3. Store prediction.
        
        train_series = train_df['Return_Unscaled']
        test_series = map(float, test_df['Return_Unscaled'].values) # Iterator
        
        # Force 'c' (constant/intercept) to avoid (0,0,0) no-intercept model which yields DA=0
        # This ensures the model predicts at least the mean return (Drift) rather than pure 0.
        arima_model = auto_arima(train_series, seasonal=False, trend='c', trace=True, error_action='ignore', suppress_warnings=True)
        print(f"  Best Order: {arima_model.order}")
        
        predictions = []
        # Rolling Forecast
        print("  Running Rolling Forecast for ARIMA...")
        
        # We need to predict ONE step ahead of the training set first?
        # NO. test_df row 0 target is R_{T+1}.
        # We have R_T available in test_df['Return_Unscaled'] row 0.
        # So we should update with that first?
        # Let's check:
        # y_test[0] is Target at index T -> R_{T+1}.
        # We want to predict R_{T+1}.
        # We verify we have R_T.
        # Yes, test_df.iloc[0]['Return_Unscaled'] is R_T.
        
        for obs in test_series:
            # Update with the new observation (Realizes R_t)
            arima_model.update(obs)
            # Predict 1 step ahead (Forecast R_{t+1})
            pred = arima_model.predict(n_periods=1)[0]
            predictions.append(pred)
            
        predictions = np.array(predictions)
        
        rmse, mae, da = eval_metrics(y_test, predictions)
        
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  Directional Accuracy: {da}")
        
        mlflow.log_param("order", str(arima_model.order))
        mlflow.log_param("seasonal_order", str(arima_model.seasonal_order))
        mlflow.log_param("aic", arima_model.aic())
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("directional_accuracy", da)
        # Verify if signature is needed/possible for ARIMA. Usually not critical for basic stats models unless serving.
        # Log Feature Config
        feature_config = {
            "features": ['Return_Unscaled'],
            "target": target_col,
            "model_type": "ARIMA"
        }
        mlflow.log_dict(feature_config, "feature_config.json")

        mlflow.sklearn.log_model(arima_model, artifact_path="model")

    # --- Model C: LSTM ---
    with mlflow.start_run(run_name="LSTM") as run:
        mlflow.set_tags(tags)
        mlflow.set_tag("model_type", "LSTM")
        
        print("Training LSTM...")
        
        # --- NEW LSTM IMPLEMENTATION (Sliding Window & Deeper Arch) ---
        
        # Helper to create sequences
        def create_sequences(data, target, time_steps):
            X, y = [], []
            # We need 'time_steps' of history to predict target at 'i + time_steps - 1'
            # (which matches the feature row at that index)
            # Alignment:
            # Inputs: rows [i : i+time_steps] (0..59)
            # Target: row [i+time_steps-1] (59)
            # Wait, row 59 has Target_59 which is Return_{60}.
            # We want to predict Return_{60} using Features_{0}..Features_{59}.
            # So X = data[i: i+time_steps]
            # y = target[i+time_steps-1]
            for i in range(len(data) - time_steps + 1):
                X.append(data[i:(i + time_steps)])
                val = target[i + time_steps - 1]
                y.append(val)
            return np.array(X), np.array(y)

        # Config
        time_steps = 60
        n_features = X_train.shape[1]
        
        # Generate Sequences
        # Note: We must use .values
        X_train_seq, y_train_seq = create_sequences(X_train.values, y_train.values, time_steps)
        X_test_seq, y_test_seq = create_sequences(X_test.values, y_test.values, time_steps)
        
        print(f"  LSTM Input Shape: {X_train_seq.shape}")
        
        epochs = 20
        batch_size = 32
        
        model = Sequential()
        model.add(Input(shape=(time_steps, n_features)))
        
        # Layer 1
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.2))
        
        # Layer 2
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.2))
        
        # Layer 3
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        
        # Layer 4
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Output
        model.add(Dense(25))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train
        model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size, verbose=1)
        
        predictions = model.predict(X_test_seq)
        predictions = predictions.flatten()
        
        rmse, mae, da = eval_metrics(y_test_seq, predictions)
        
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  Directional Accuracy: {da}")

        # --- OLD LSTM IMPLEMENTATION (Commented for comparison) ---
        """
        # Input Shape: [Samples, Time Steps, Features]
        # We treat each row as 1 Time Step with N features.
        time_steps = 1
        n_features = X_train.shape[1]
        
        X_train_reshaped = X_train.values.reshape((X_train.shape[0], time_steps, n_features))
        X_test_reshaped = X_test.values.reshape((X_test.shape[0], time_steps, n_features))
        
        epochs = 20
        batch_size = 32
        
        model = Sequential()
        model.add(Input(shape=(time_steps, n_features)))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train
        model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        predictions = model.predict(X_test_reshaped)
        predictions = predictions.flatten()
        
        rmse, mae, da = eval_metrics(y_test, predictions)

        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  Directional Accuracy: {da}")
        """
        # -------------------------------------------------------------
        
        # Log Architecture Details Modularly
        mlflow.log_param("input_shape", f"[{X_train_seq.shape[0]}, {time_steps}, {n_features}]")
        mlflow.log_param("time_steps", time_steps)
        mlflow.log_param("n_features", n_features)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        
        # Log Layer Details
        model_summary = []
        for i, layer in enumerate(model.layers):
            layer_config = layer.get_config()
            layer_name = layer_config['name']
            layer_class = layer.__class__.__name__
            units = layer_config.get('units', 'N/A')
            activation = layer_config.get('activation', 'N/A')
            
            mlflow.log_param(f"layer_{i}_class", layer_class)
            mlflow.log_param(f"layer_{i}_units", units)
            mlflow.log_param(f"layer_{i}_activation", activation)
            
            model_summary.append(f"{layer_class}(units={units}, activation={activation})")
            
        mlflow.log_param("model_arch_summary", " -> ".join(model_summary))
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("directional_accuracy", da)
        
        # Log Scaler as Artifact
        mlflow.log_artifact(SCALER_PATH, artifact_path="scaler")
        
        # Log Feature Config
        feature_config = {
            "features": feature_cols,
            "target": target_col,
            "time_steps": time_steps,
            "n_features": n_features,
            "model_type": "LSTM"
        }
        mlflow.log_dict(feature_config, "feature_config.json")

        # Infer and log signature (This resolves the TF warning)
        signature = infer_signature(X_train_seq, predictions)
        mlflow.tensorflow.log_model(model, artifact_path="model", signature=signature)

if __name__ == "__main__":
    train_models()
