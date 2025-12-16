import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import os
import joblib
import logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pmdarima import auto_arima
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MLFLOW_EXPERIMENT_NAME = "EURUSD_Prediction_Comparison"

def load_data(processed_dir="data/processed"):
    train = pd.read_csv(f"{processed_dir}/train.csv", index_col="Date", parse_dates=True)
    test = pd.read_csv(f"{processed_dir}/test.csv", index_col="Date", parse_dates=True)
    return train, test

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# -------------------------------------------------------------------
# 1. Linear Regression
# -------------------------------------------------------------------
def train_linear_regression(train, test, features, target):
    logging.info(f"Training Linear Regression for {target}...")
    
    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]
    
    with mlflow.start_run(run_name=f"LinearRegression_{target}"):
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        rmse, mae, r2 = eval_metrics(y_test, predictions)
        
        logging.info(f"LR {target} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, f"model_lr_{target}")
        
        # Save locally for easy access
        os.makedirs("models/linear_regression", exist_ok=True)
        joblib.dump(model, f"models/linear_regression/model_{target}.pkl")
        
        return rmse

# -------------------------------------------------------------------
# 2. Auto ARIMA (Univariate)
# -------------------------------------------------------------------
def train_auto_arima(train, test, target):
    logging.info(f"Training Auto ARIMA for {target}...")
    
    # ARIMA is univariate, uses only the target history
    y_train = train[target]
    y_test = test[target]
    
    with mlflow.start_run(run_name=f"AutoARIMA_{target}"):
        # Stepwise search to minimize AIC
        # We assume data is stationary or auto_arima handles differencing (d)
        model = auto_arima(y_train, start_p=1, start_q=1,
                           max_p=5, max_q=5, m=1, # Non-seasonal for daily data often better
                           start_P=0, seasonal=False, 
                           d=None, trace=False,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
        
        logging.info(f"Best ARIMA order: {model.order}")
        
        # Forecast
        predictions, conf_int = model.predict(n_periods=len(y_test), return_conf_int=True)
        # Predictions is a pd.Series or array matching test length
        
        rmse, mae, r2 = eval_metrics(y_test, predictions)
        
        logging.info(f"ARIMA {target} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        mlflow.log_params({"order": str(model.order)})
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Save locally
        os.makedirs("models/arima", exist_ok=True)
        joblib.dump(model, f"models/arima/model_{target}.pkl")
        
        return rmse

# -------------------------------------------------------------------
# 3. LSTM
# -------------------------------------------------------------------
def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_lstm(train, test, features, target):
    logging.info(f"Training LSTM for {target}...")
    
    time_steps = 10 # Lookback window
    
    # Scale data? LSTM is sensitive to scale. 
    # For simplicity, we assume Return is small (-0.01 to 0.01), Price is ~1.0-1.2.
    # Ideally we use scalers, but for this "simple" pass we might skip if variables are similar ranges.
    # However, Returns and Price are very different scales. 
    # We will fit separate LSTMs, so scaling is per-model. 
    # Let's Skip explicit scaling for Returns (already small), but Price might need it? 
    # Actually, Price variation is small too (1.0 to 1.2), so usually okay without standard scaling for a basic demo.
    
    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]
    
    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)
    
    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        logging.warning("Not enough data for LSTM sequences.")
        return 999
        
    with mlflow.start_run(run_name=f"LSTM_{target}"):
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(time_steps, len(features))))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
        # Train
        model.fit(X_train_seq, y_train_seq, epochs=20, batch_size=32, verbose=0)
        
        predictions = model.predict(X_test_seq)
        
        # y_test_seq corresponds to y_test[time_steps:]
        real_y = y_test_seq
        
        rmse, mae, r2 = eval_metrics(real_y, predictions.flatten())
        
        logging.info(f"LSTM {target} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Save locally - Keras format
        os.makedirs("models/lstm", exist_ok=True)
        model.save(f"models/lstm/model_{target}.keras")
        
        return rmse

def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    train, test = load_data()
    
    # Define features and targets
    # Features created in step 02: Close_Lag_1..5, Return_Lag_1..5, SMA_7, SMA_30
    feature_cols = [c for c in train.columns if 'Lag' in c or 'SMA' in c]
    
    logging.info(f"Features: {feature_cols}")
    
    targets = ['Target_Return', 'Target_Close']
    
    results = {}
    
    for target in targets:
        logging.info(f"--- Processing Target: {target} ---")
        
        # Linear Regression
        rmse_lr = train_linear_regression(train, test, feature_cols, target)
        
        # ARIMA (Only makes sense for Price/Close usually, or Return series itself)
        # Note: Auto ARIMA takes the univariate series. 
        # If target is 'Target_Return', we just fit on 'Return' history? 
        # Actually we fit on 'Target_Return' variable of train set? 
        # No, ARIMA models the series itself. The Target_Return column IS the Return series shifted.
        # So we should use the UNshifted 'Return' or 'Close' to train, and predict forward?
        # Simpler: Use the 'Target' column as the series we want to model, knowing it aligns with features.
        # Strict time series definition: y_t. 
        rmse_arima = train_auto_arima(train, test, target)
        
        # LSTM
        rmse_lstm = train_lstm(train, test, feature_cols, target)
        
        results[target] = {
            "LinearRegression": rmse_lr,
            "ARIMA": rmse_arima,
            "LSTM": rmse_lstm
        }

    logging.info("--- Experiment Results (RMSE) ---")
    print(results)
    
    # Simple logic to identify best model
    best_models = {}
    for target, scores in results.items():
        best_model_name = min(scores, key=scores.get)
        best_models[target] = best_model_name
        logging.info(f"Best model for {target}: {best_model_name} (RMSE: {scores[best_model_name]:.4f})")

    # TODO: In a real system, we'd package the best model here.
    # For now, models are saved in models/

if __name__ == "__main__":
    main()
