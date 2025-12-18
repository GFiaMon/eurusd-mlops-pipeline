"""
Quick test to verify the Flask app can load and predict with the Linear Regression model
"""
import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

load_dotenv()

# Import the app components
from api.app_cloud import model, scaler, feature_config, load_success, FeatureEngineer, df_data

print("=" * 60)
print("FLASK APP LOAD TEST")
print("=" * 60)

print(f"\n✓ Model loaded: {load_success}")
if load_success:
    print(f"✓ Model type: {feature_config.get('model_type', 'Unknown')}")
    print(f"✓ Features: {feature_config.get('features', [])}")
    print(f"✓ Scaler available: {scaler is not None}")
    print(f"✓ Data loaded: {df_data is not None}")
    
    if df_data is not None:
        print(f"✓ Data shape: {df_data.shape}")
        
        # Test prediction
        print("\n" + "=" * 60)
        print("TESTING PREDICTION")
        print("=" * 60)
        
        fe = FeatureEngineer(feature_config, scaler)
        X, current_price = fe.preprocess(df_data)
        
        print(f"\n✓ Preprocessed data type: {type(X)}")
        print(f"✓ Preprocessed data shape: {X.shape}")
        if isinstance(X, pd.DataFrame):
            print(f"✓ Column names: {list(X.columns)}")
        print(f"✓ Current price: {current_price:.4f}")
        
        # Make prediction
        try:
            pred_return = model.predict(X)
            print(f"\n✓ Prediction successful!")
            print(f"✓ Predicted return: {pred_return}")
            
            # Handle varying output shapes
            import numpy as np
            if hasattr(pred_return, 'values'):
                pred_return = pred_return.values
            if isinstance(pred_return, (list, np.ndarray)) and len(np.shape(pred_return)) > 0:
                pred_return = pred_return.flatten()[0]
            else:
                pred_return = float(pred_return)
                
            predicted_price = current_price * (1 + pred_return)
            change_pct = pred_return * 100
            
            print(f"✓ Predicted price: ${predicted_price:.4f}")
            print(f"✓ Change: {change_pct:+.2f}%")
            
            print("\n" + "=" * 60)
            print("✅ ALL TESTS PASSED - Flask app is ready!")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            print("\n" + "=" * 60)
            print("❌ TEST FAILED")
            print("=" * 60)
            sys.exit(1)
else:
    print("\n❌ Model not loaded - check MLflow connection")
    sys.exit(1)
