import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load Processed Data
TRAIN_DATA_PATH = os.path.join("data", "processed", "train.csv")
TEST_DATA_PATH = os.path.join("data", "processed", "test.csv")

def verify_data():
    if not os.path.exists(TRAIN_DATA_PATH):
        print("Data not found. Run pipeline first.")
        return

    print("Loading data...")
    train_df = pd.read_csv(TRAIN_DATA_PATH, index_col=0, parse_dates=True)
    test_df = pd.read_csv(TEST_DATA_PATH, index_col=0, parse_dates=True)
    
    # 1. Correlation Check
    print("\n--- 1. Correlation Check (Leakage Detection) ---")
    # We check if any feature is suspiciously correlated with 'Target' (> 0.95)
    # Target is prediction for Tomorrow. Features are info from Today.
    # High correlation is GOOD for prediction, but Perfect (1.0) usually means we accidentally used Tomorrow's data.
    
    validation_df = train_df.copy()
    corr_matrix = validation_df.corr()
    target_corrs = corr_matrix['Target'].sort_values(ascending=False)
    
    print("Correlations with Target:")
    print(target_corrs)
    
    suspicious = target_corrs[target_corrs > 0.95]
    suspicious = suspicious.drop('Target') # Remove self
    
    if not suspicious.empty:
        print(f"\n⚠️ WARNING: Suspicious correlations found! Possible Leakage:\n{suspicious}")
    else:
        print("\n✅ No suspicious correlations (>0.95) detected.")

    # 2. Baseline Metrics (Predicting 0.0)
    print("\n--- 2. Baseline Verification ---")
    y_test = test_df['Target']
    y_pred_zero = np.zeros_like(y_test)
    
    rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_zero))
    mae_base = mean_absolute_error(y_test, y_pred_zero)
    print(f"Baseline (Predict 0): RMSE={rmse_base:.5f}, MAE={mae_base:.5f}")
    
    # 3. Linear Regression Verification
    print("\n--- 3. Linear Regression Weights ---")
    cols = [c for c in train_df.columns if c not in ['Target', 'Return_Unscaled']]
    lr = LinearRegression()
    lr.fit(train_df[cols], train_df['Target'])
    
    importance = pd.Series(lr.coef_, index=cols).sort_values(key=abs, ascending=False)
    print("Feature Importance (Coefficients):")
    print(importance.head(10))
    
    # Check DA again
    pred = lr.predict(test_df[cols])
    da = np.mean(np.sign(pred) == np.sign(y_test))
    print(f"\nRecalculated Linear Regression DA: {da:.4f}")

if __name__ == "__main__":
    verify_data()
