# Model Selection Flow in Flask API

## How the Flask App Gets the "Right Model"

The system uses a **two-layer approach** to determine which model to load:

### Layer 1: best_model_info.json (The Pointer)
**Location:** `models/best_model_info.json` (also uploaded to S3)

This JSON file acts as a **pointer** that tells the Flask app:
- Which model to load (`model_uri`)
- What type of model it is (`model_type`)
- Which MLflow alias to use (`model_alias`)

**Example:**
```json
{
    "model_uri": "models:/EURUSD_LinearRegression@champion",
    "model_type": "LinearRegression",
    "model_alias": "champion"
}
```

### Layer 2: MLflow Model Registry (The Source of Truth)
The `model_uri` contains the **@champion** alias, which MLflow resolves to the actual model version.

---

## The Complete Flow

```
Docker Container Starts
        ↓
1. Load best_model_info.json
   - From S3 (if USE_S3=true)
   - OR from local models/ folder
        ↓
2. Read model_uri: "models:/EURUSD_LinearRegression@champion"
        ↓
3. Contact MLflow Server
   - Resolve @champion alias
   - Get actual model version
        ↓
4. Detect model_type: "LinearRegression"
        ↓
5. Load model with appropriate loader
   - ARIMA → mlflow.sklearn.load_model()
   - Others → mlflow.pyfunc.load_model()
        ↓
6. Download artifacts from MLflow
   - feature_config.json
   - scaler.pkl
        ↓
7. Initialize FeatureEngineer with config
   - Knows which features to use
   - Knows if it's LSTM (needs 3D reshape)
        ↓
8. Flask App Ready to Serve
```

---

## Two Ways to Change the Champion Model

### Method 1: Manual Change in MLflow UI (Your Question!)

**Steps:**
1. Go to MLflow UI → Models → Select a model (e.g., EURUSD_LSTM)
2. Find a version and set alias to "champion"
3. **IMPORTANT:** This changes the MLflow registry, BUT...
4. The Flask app still loads based on `best_model_info.json`!

**What happens:**
- If `best_model_info.json` says: `"model_uri": "models:/EURUSD_LinearRegression@champion"`
- The app will STILL try to load from EURUSD_LinearRegression model (not LSTM!)
- Even if you changed the LSTM model's alias to "champion"

**Why?** Because the model NAME is hardcoded in the JSON file!

### Method 2: Automated Change via Evaluation Script (Recommended)

**Steps:**
1. Run: `python src/04_evaluate_select.py`
2. Script evaluates all models
3. Ranks them by RMSE
4. Assigns aliases automatically:
   - Best model → @champion
   - 2nd best → @challenger  
   - 3rd best → @candidate
5. **Updates best_model_info.json** with the champion's details
6. Uploads updated JSON to S3

**What happens:**
- `best_model_info.json` is updated to point to the new champion
- Next Docker restart will load the new champion automatically

---

## Current Limitation & Solution

### ❌ Current Limitation:
If you manually change the @champion alias in MLflow UI, the Flask app won't automatically pick it up because `best_model_info.json` still points to the old model name.

### ✅ Solution Option 1: Make it Fully Dynamic
We can modify the Flask app to:
1. Query MLflow for ALL models with @champion alias
2. Load whichever model has that alias (regardless of name)
3. Ignore the model_name in best_model_info.json

### ✅ Solution Option 2: Keep Current Workflow
Always use the evaluation script to promote models. This ensures:
- Consistent model selection based on metrics
- Proper tracking and documentation
- S3 and local files stay in sync

---

## What You Asked About

> "if i change it manually on the ui and then reload the docker it should change the preprocessing and model?"

**Current Answer:** ❌ No, not automatically.

**Reason:** The `best_model_info.json` file contains:
```json
"model_uri": "models:/EURUSD_LinearRegression@champion"
```

This hardcodes the model NAME (EURUSD_LinearRegression). Even if you give @champion to LSTM in the UI, the Flask app will still try to load from the LinearRegression model.

**To make it work:** You'd need to either:
1. Manually edit `best_model_info.json` to change the model_name
2. OR modify the Flask app to search for @champion across all models

---

## Recommendation

Would you like me to implement **Solution Option 1** to make the system fully dynamic? 

This would allow you to:
1. Change @champion in MLflow UI
2. Restart Docker
3. New model loads automatically

The trade-off is that you lose the explicit tracking in `best_model_info.json`, but you gain flexibility.
