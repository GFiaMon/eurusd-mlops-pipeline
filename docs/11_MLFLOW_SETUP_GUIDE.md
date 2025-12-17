# MLflow Setup and Model Registry Guide

## Table of Contents
1. [Overview](#overview)
2. [Why SQLite Backend?](#why-sqlite-backend)
3. [Model Registry Strategy](#model-registry-strategy)
4. [Implementation Details](#implementation-details)
5. [Running the Pipeline](#running-the-pipeline)
6. [Viewing Results in MLflow UI](#viewing-results-in-mlflow-ui)
7. [Deployment Strategy](#deployment-strategy)
8. [Common Issues and Solutions](#common-issues-and-solutions)

---

## Overview

This project uses **MLflow** for experiment tracking and model registry. We track three different model types:
- **Linear Regression**
- **ARIMA**
- **LSTM**

The system automatically ranks them by performance (RMSE) and assigns semantic aliases for easy deployment.

---

## Why SQLite Backend?

### The Problem with File-Based Tracking
MLflow can use a simple file-based backend (`./mlruns` directory), which works for basic experiment tracking. However, the **Model Registry** features (versioning, aliasing, stage management) **require a database backend**.

### The Solution
We use a local **SQLite database** (`mlflow.db`) which enables:
- ✅ Model versioning
- ✅ Model aliasing (`@champion`, `@challenger`, `@candidate`)
- ✅ Centralized model metadata
- ✅ Easy migration to production databases (PostgreSQL, MySQL)

### Configuration
Both training and evaluation scripts set the tracking URI:
```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

---

## Model Registry Strategy

### Individual Model Registration
Each model type is registered separately:
- `EURUSD_LinearRegression`
- `EURUSD_ARIMA`
- `EURUSD_LSTM`

This provides:
- **Provenance**: Full history of each architecture
- **Comparison**: Easy to see how each model type evolves over time
- **Flexibility**: Can deploy any specific architecture if needed

### Alias-Based Ranking System
Instead of using deprecated "Stages" (Production, Staging, Archived), we use **Model Aliases**:

| Rank | Alias | Meaning |
|------|-------|---------|
| 1st | `@champion` | Best performing model - ready for production |
| 2nd | `@challenger` | Second best - candidate for A/B testing |
| 3rd | `@candidate` | Third best - backup option |

**Key Advantages:**
- ✅ Modern MLflow standard (Stages deprecated since v2.9.0)
- ✅ Flexible: Can have multiple aliases per model
- ✅ Clear deployment target: Always load `@champion`
- ✅ Easy A/B testing: Compare `@champion` vs `@challenger`

### Why We Avoided the "Global Model" Approach
**Initial Attempt (Rejected):**
We initially tried creating a single `EURUSD_Production_Winner` model that would contain versions from all three architectures (Linear v1, LSTM v2, ARIMA v3, etc.).

**Problems:**
- ❌ Confusing version history (mixing different architectures)
- ❌ Created duplicate versions on every script run
- ❌ Harder to understand which version corresponds to which architecture
- ❌ Messy registry with 6+ versions after just 2 runs

**Final Solution (Adopted):**
Keep models separate, assign aliases directly to the specific models that earned them.

---

## Implementation Details

### Training Script (`src/03_train_models.py`)

**Key Components:**
1. **Database Connection:**
   ```python
   mlflow.set_tracking_uri("sqlite:///mlflow.db")
   mlflow.set_experiment(EXPERIMENT_NAME)
   ```

2. **Metric Logging:**
   - RMSE (primary ranking metric)
   - MAE (Mean Absolute Error)
   - Directional Accuracy (custom metric for financial forecasting)

3. **Model Logging:**
   - Uses `mlflow.sklearn.log_model()` for Linear/ARIMA
   - Uses `mlflow.tensorflow.log_model()` for LSTM
   - Includes model signatures for input/output validation

4. **Metadata Tagging:**
   ```python
   tags = {
       "developer": "User",
       "project": "EURUSD_Capstone",
       "data_version": "v1"
   }
   mlflow.set_tag("model_type", "LinearRegression")  # or "ARIMA", "LSTM"
   ```

### Evaluation Script (`src/04_evaluate_select.py`)

**Workflow:**
1. **Load Test Data** → Calculate naive baseline
2. **Query MLflow Runs** → Find best run per model type
3. **Register Models** → Create versioned entries in registry
4. **Rank & Alias** → Sort by RMSE, assign `@champion`, `@challenger`, `@candidate`
5. **Save Deployment Info** → Write `models/best_model_info.json`

**Key Logic:**
```python
# Sort by RMSE (ascending: lower is better)
ranked_models = sorted(registered_models, key=lambda x: x['rmse'])

# Assign aliases
for rank, model_info in enumerate(ranked_models):
    alias = alias_map[rank]  # champion, challenger, candidate
    client.set_registered_model_alias(
        name=model_info['name'],
        alias=alias,
        version=model_info['version']
    )
```

---

## Running the Pipeline

### Step 1: Train Models
```bash
python src/03_train_models.py
```

**Output:**
- Creates/updates `mlflow.db`
- Logs 3 runs (Linear, ARIMA, LSTM) to experiment `EURUSD_Experiments`
- Saves model artifacts to `mlruns/<experiment_id>/<run_id>/artifacts/model/`

### Step 2: Evaluate & Register
```bash
python src/04_evaluate_select.py
```

**Output:**
- Registers best version of each model type
- Assigns aliases based on performance ranking
- Creates `models/best_model_info.json` with deployment URI

**Example Output:**
```
*** CHAMPIONSHIP RANKING (Lower RMSE is Better) ***
1. LSTM (RMSE: 0.00510) -> EURUSD_LSTM (v1) gets alias '@champion'
2. LinearRegression (RMSE: 0.00511) -> EURUSD_LinearRegression (v1) gets alias '@challenger'
3. ARIMA (RMSE: 0.00512) -> EURUSD_ARIMA (v1) gets alias '@candidate'

Deployment Choice: EURUSD_LSTM (v1)
Validation: Champion beats Baseline (0.00510 < 0.00716)
```

### Step 3: View in MLflow UI
```bash
# Use port 5001 to avoid macOS AirPlay conflicts on port 5000
PYTHONWARNINGS="ignore" mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Open browser: `http://localhost:5001`

---

## Viewing Results in MLflow UI

### Experiments Tab
- View all runs for `EURUSD_Experiments`
- Compare metrics (RMSE, MAE, Directional Accuracy)
- Sort by any metric to find best performers
- View parameters, tags, and artifacts

### Models Tab
Navigate to see registered models:
- `EURUSD_LinearRegression`
- `EURUSD_ARIMA`
- `EURUSD_LSTM`

**For each model:**
- Click to see version history
- View assigned aliases (champion/challenger/candidate)
- See which run each version came from
- Download model artifacts

---

## Deployment Strategy

### Loading the Champion Model

**Option 1: Using the JSON File (Recommended for Automation)**
```python
import json
import mlflow

# Load deployment info
with open('models/best_model_info.json', 'r') as f:
    info = json.load(f)

# Load the champion model
model_uri = info['model_uri']  # e.g., "models:/EURUSD_LSTM@champion"
model = mlflow.pyfunc.load_model(model_uri)

# Make predictions
predictions = model.predict(X_test)
```

**Option 2: Direct URI (If you know the champion)**
```python
import mlflow

# Always loads the current champion, regardless of which model type it is
model = mlflow.pyfunc.load_model("models:/EURUSD_LSTM@champion")
```

### A/B Testing
```python
# Load champion and challenger for comparison
champion = mlflow.pyfunc.load_model("models:/EURUSD_LSTM@champion")
challenger = mlflow.pyfunc.load_model("models:/EURUSD_LinearRegression@challenger")

# Route 90% traffic to champion, 10% to challenger
if random.random() < 0.9:
    prediction = champion.predict(X)
else:
    prediction = challenger.predict(X)
```

---

## Common Issues and Solutions

### Issue 1: Port 5000 Already in Use
**Error:**
```
[ERROR] Connection in use: ('127.0.0.1', 5000)
```

**Cause:** macOS AirPlay Receiver uses port 5000 by default.

**Solution:**
```bash
# Use a different port
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

**Or kill zombie processes:**
```bash
lsof -i :5000
kill <PID>
```

---

### Issue 2: Deprecation Warning for Stages
**Error:**
```
FutureWarning: mlflow.tracking.client.MlflowClient.transition_model_version_stage is deprecated
```

**Cause:** Using old "Stages" API (Production, Staging, Archived).

**Solution:** Use aliases instead:
```python
# ❌ OLD (Deprecated)
client.transition_model_version_stage(name="MyModel", version=1, stage="Production")

# ✅ NEW (Modern)
client.set_registered_model_alias(name="MyModel", alias="champion", version=1)
```

---

### Issue 3: Multiple Versions Created on Re-runs
**Problem:** Running `04_evaluate_select.py` multiple times creates v1, v2, v3... for the same run.

**Current Behavior:** This is expected. Each evaluation creates new versions.

**Future Enhancement (Optional):**
Add deduplication logic:
```python
# Check if run already registered
existing = client.search_model_versions(f"name='{model_name}' and run_id='{run_id}'")
if existing:
    version = existing[0].version
else:
    reg_model = mlflow.register_model(model_uri, model_name)
    version = reg_model.version
```

---

### Issue 4: `pkg_resources` Deprecation Warning
**Warning:**
```
UserWarning: pkg_resources is deprecated as an API
```

**Cause:** MLflow internal dependency issue with newer Setuptools.

**Solution:** Already suppressed in code:
```python
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
```

For MLflow UI, use:
```bash
PYTHONWARNINGS="ignore" mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

---

### Issue 5: Starting Fresh / Clearing All Data
**When to do this:**
- Testing new configurations
- Registry became too messy
- Want to reset version numbers

**How:**
```bash
# Delete database and file-based runs
rm mlflow.db
rm -rf mlruns

# Re-run training and evaluation
python src/03_train_models.py
python src/04_evaluate_select.py
```

**⚠️ Warning:** This deletes ALL experiment history and registered models.

---

## Best Practices

### 1. Consistent Metric Naming
Always log the same metrics across all models:
```python
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("mae", mae)
mlflow.log_metric("directional_accuracy", da)
```

### 2. Meaningful Tags
Use tags for filtering and organization:
```python
mlflow.set_tag("model_type", "LSTM")
mlflow.set_tag("developer", "YourName")
mlflow.set_tag("data_version", "v1")
```

### 3. Model Signatures
Always log signatures for production models:
```python
from mlflow.models import infer_signature
signature = infer_signature(X_train, predictions)
mlflow.sklearn.log_model(model, "model", signature=signature)
```

### 4. Version Control Your Scripts
The MLflow database tracks model versions, but your training scripts should be version-controlled (Git) to ensure reproducibility.

### 5. Document Experiments
Use MLflow's description fields:
```python
client.update_registered_model(
    name="EURUSD_LSTM",
    description="LSTM model for EUR/USD next-day return prediction. Uses 5-day MA and 5 lag features."
)
```

---

## Migration to Production

### Local Development (Current)
```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

### Production (AWS/Cloud)
```python
# Option 1: Remote MLflow Server
mlflow.set_tracking_uri("http://mlflow-server.example.com:5000")

# Option 2: AWS S3 + PostgreSQL
mlflow.set_tracking_uri("postgresql://user:pass@db.example.com/mlflow")
os.environ['MLFLOW_ARTIFACT_ROOT'] = 's3://my-bucket/mlflow-artifacts'
```

**Benefits of Remote Setup:**
- ✅ Team collaboration
- ✅ Centralized model registry
- ✅ Scalable artifact storage
- ✅ Production-grade database

---

## Summary

**What We Built:**
1. ✅ SQLite-backed MLflow tracking for Model Registry support
2. ✅ Separate model registration for each architecture (Linear, ARIMA, LSTM)
3. ✅ Modern alias-based ranking system (`@champion`, `@challenger`, `@candidate`)
4. ✅ Automated evaluation and selection pipeline
5. ✅ Deployment-ready JSON file with champion URI

**Key Takeaways:**
- Always use a database backend (SQLite minimum) for Model Registry
- Use aliases instead of deprecated stages
- Keep model families separate for clarity
- Rank and tag based on business metrics (RMSE in our case)
- Automate the evaluation → registration → aliasing workflow

**Deployment URI:**
```
models:/EURUSD_<ModelType>@champion
```
This URI dynamically points to the best model, making deployment code simple and stable.
