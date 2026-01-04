# Flask API Update - FAQ & Detailed Guide

## 1. MLflow Version Compatibility

**Q: The server has MLflow 3.7.0 and AWS works with MLflow 3.0. Should I use mlflow >=3.0.0?**

**A: Yes, you should upgrade!** Here's why:

- Your AWS MLflow server is running 3.7.0
- The API client should match or be compatible with the server version
- MLflow 3.x has better compatibility and features

**Action Required:**
Update [`api/requirements.txt`](file:///Users/guillermo/Documents/Ironhack/M8_Capstone/2-Capstone-2_MLOps/1-Development/eurusd-capstone/api/requirements.txt):
```diff
- mlflow==2.11.3
+ mlflow>=3.0.0
```

---

## 2. Docker Changes Explained

**Q: What did you change in Docker?**

**Before:**
```dockerfile
# Copy models directory
COPY models/ /app/models/
```

**After:**
```dockerfile
# We do NOT copy models/ because we fetch best_model_info.json 
# and model artifacts from S3/MLflow at runtime.
# This ensures the container serves the latest champion without rebuild.

# Create necessary directories
RUN mkdir -p /app/models /app/data/processed /app/data/raw
```

**Why this matters:**
- **Before**: Model files were baked into the Docker image → Required rebuilding the image every time you trained a new model
- **After**: The container downloads the latest champion model from MLflow/S3 when it starts → No rebuild needed, just restart the container

**Analogy**: It's like changing from a hardcoded phone number to looking it up in a contact list. You can update the contact without changing the phone app.

---

## 3. How Model Switching Works

**Q: How does the API automatically adapt to new champion models? How often does it check? What if the architecture changes?**

**A: The API checks ONCE at startup, not continuously.**

### How It Works:

1. **Container Starts** → API runs `load_from_mlflow()`
2. **Fetches Config** → Downloads `best_model_info.json` from S3 (or local)
3. **Reads Champion Info**:
   ```json
   {
     "model_uri": "models:/EURUSD_LSTM@champion",
     "model_type": "LSTM"
   }
   ```
4. **Loads Model** → Uses MLflow to load whatever model has the `@champion` alias
5. **Adapts Features** → Uses the feature config from that model's artifacts

### When Does It Switch?

**NOT automatic during runtime.** You must:
1. Train new models
2. Run `src/04_evaluate_select.py` (assigns `@champion` alias to best model)
3. **Restart the API container** on EC2

### Different Architecture Support:

The API handles different model types through the `FeatureEngineer` class:

```python
# For LSTM (needs reshaping)
if 'time_steps' in config:
    X = X.reshape((1, time_steps, n_features))

# For Linear Regression/ARIMA (no reshaping)
else:
    X = scaled_features
```

**Supported architectures:**
- ✅ Linear Regression
- ✅ ARIMA
- ✅ LSTM (with time_steps=1)

**If you add a new architecture** (e.g., GRU, Transformer):
- You may need to update the `FeatureEngineer.preprocess()` method
- Add architecture-specific reshaping logic

---

## 4. Adding New Features

**Q: What if I add features like volatility? Do I have to change the code?**

**A: Partially automated, but requires some manual updates.**

### Steps to Add New Features:

#### 1. Update Preprocessing ([`src/02_preprocess.py`](file:///Users/guillermo/Documents/Ironhack/M8_Capstone/2-Capstone-2_MLOps/1-Development/eurusd-capstone/src/02_preprocess.py))
```python
# Add your new feature
df['Volatility'] = df['Return'].rolling(window=10).std()

# Update feature list
feature_cols = ['Return', 'MA_5', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'Volatility']
```

#### 2. Update Training Script ([`src/03_train_models.py`](file:///Users/guillermo/Documents/Ironhack/M8_Capstone/2-Capstone-2_MLOps/1-Development/eurusd-capstone/src/03_train_models.py))
```python
feature_cols = ['Return', 'MA_5', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'Volatility']
```

#### 3. Update API Feature Engineering ([`api/app_cloud.py`](file:///Users/guillermo/Documents/Ironhack/M8_Capstone/2-Capstone-2_MLOps/1-Development/eurusd-capstone/api/app_cloud.py))
```python
class FeatureEngineer:
    def preprocess(self, df):
        # ... existing code ...
        
        # Add your new feature calculation
        data['Volatility'] = data['Return'].rolling(window=10).std()
        
        # The feature selection is automatic from config
        last_row = data.iloc[[-1]][self.feature_cols]
```

**What's Automatic:**
- Feature selection (uses `self.feature_cols` from config)
- Scaling (applies to all features in the list)

**What's Manual:**
- Feature calculation logic (you must add the calculation in both preprocessing and API)

**Future Improvement Idea:**
You could create a shared `feature_engineering.py` module that both training and API import.

---

## 5. S3 Configuration

**Q: Where do you set USE_S3=true? Where is this code?**

**A: It's set via environment variables, not hardcoded.**

### In the Code ([`api/app_cloud.py`](file:///Users/guillermo/Documents/Ironhack/M8_Capstone/2-Capstone-2_MLOps/1-Development/eurusd-capstone/api/app_cloud.py), lines 40-44):
```python
USE_S3 = os.getenv('USE_S3', 'false').lower() == 'true'
S3_BUCKET = os.getenv('S3_BUCKET', 'your-bucket-name')
S3_MODEL_INFO_KEY = os.getenv('S3_MODEL_INFO_KEY', 'models/best_model_info.json')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
```

### How to Set (3 Options):

#### Option 1: `.env` File (Local Development)
Create/edit `.env` in project root:
```bash
USE_S3=true
S3_BUCKET=eurusd-ml-models
MLFLOW_TRACKING_URI=http://3.235.184.216:5000
AWS_REGION=us-east-1
```

#### Option 2: Docker Run Command (EC2 Deployment)
```bash
docker run -e USE_S3=true \
           -e S3_BUCKET=eurusd-ml-models \
           -e MLFLOW_TRACKING_URI=http://3.235.184.216:5000 \
           -e AWS_REGION=us-east-1 \
           -p 8080:8080 \
           your-image-name
```

#### Option 3: Docker Compose (Recommended for EC2)
Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  api:
    image: your-image-name
    ports:
      - "8080:8080"
    environment:
      - USE_S3=true
      - S3_BUCKET=eurusd-ml-models
      - MLFLOW_TRACKING_URI=http://3.235.184.216:5000
      - AWS_REGION=us-east-1
```

---

## 6. Environment-Based Configuration

**Q: What do you mean by "Production-Ready: Environment-based configuration"?**

**A: The same code works in different environments by changing only environment variables.**

### Example Scenarios:

| Environment | USE_S3 | MLFLOW_TRACKING_URI | Behavior |
|-------------|--------|---------------------|----------|
| **Local Dev** | `false` | `sqlite:///mlflow.db` | Uses local files |
| **Staging** | `true` | `http://staging-mlflow:5000` | Uses staging S3/MLflow |
| **Production** | `true` | `http://prod-mlflow:5000` | Uses production S3/MLflow |

**Benefits:**
- No code changes between environments
- Easy to test locally before deploying
- Secure (credentials in environment, not code)

---

## 7. EC2 Deployment Guide

**Q: What do I have to change in the EC2 instance? What code to update?**

### Step-by-Step Deployment:

#### 1. SSH into EC2
```bash
ssh -i eurusd-keypair.pem ec2-user@3.235.184.216
```

#### 2. Stop Current Container
```bash
docker ps  # Find container ID
docker stop <container-id>
docker rm <container-id>
```

#### 3. Pull Latest Code (if using Git)
```bash
cd /path/to/eurusd-capstone
git pull origin main
```

#### 4. Build New Docker Image
```bash
docker build -f Dockerfile.cloud -t eurusd-api:latest .
```

#### 5. Run Container with Environment Variables
```bash
docker run -d \
  --name eurusd-api \
  -p 8080:8080 \
  -e USE_S3=true \
  -e S3_BUCKET=eurusd-ml-models \
  -e MLFLOW_TRACKING_URI=http://localhost:5000 \
  -e AWS_REGION=us-east-1 \
  eurusd-api:latest
```

#### 6. Verify
```bash
curl http://localhost:8080/health
curl http://localhost:8080/api/predict
```

### Memory Considerations:

**Current EC2 Instance Type:** Check with `aws ec2 describe-instances`

**Recommended Minimum:**
- **t2.small** (2GB RAM) - Minimum for TensorFlow
- **t2.medium** (4GB RAM) - Recommended for production

**To Change Instance Type:**
1. Stop instance: `aws ec2 stop-instances --instance-ids i-xxx`
2. Modify: `aws ec2 modify-instance-attribute --instance-id i-xxx --instance-type t2.medium`
3. Start: `aws ec2 start-instances --instance-ids i-xxx`

---

## 8. Clean Slate - Delete All Models & Retrain

**Q: Can I delete all models and retrain from scratch?**

**A: Yes! Here's how:**

### Option 1: Via MLflow UI (Easiest)
1. Navigate to http://3.235.184.216:5000
2. Go to "Models" tab
3. For each model (EURUSD_LSTM, EURUSD_LinearRegression, EURUSD_ARIMA):
   - Click on the model
   - Delete all versions
   - Delete the model itself

### Option 2: Via Python Script
Create `scripts/clean_mlflow.py`:
```python
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://3.235.184.216:5000")
client = MlflowClient()

# Delete all registered models
for model in ["EURUSD_LSTM", "EURUSD_LinearRegression", "EURUSD_ARIMA"]:
    try:
        # Delete all versions
        versions = client.search_model_versions(f"name='{model}'")
        for v in versions:
            client.delete_model_version(model, v.version)
        # Delete model
        client.delete_registered_model(model)
        print(f"Deleted {model}")
    except Exception as e:
        print(f"Error deleting {model}: {e}")

print("All models deleted!")
```

Run it:
```bash
python scripts/clean_mlflow.py
```

### Option 3: Delete Experiment Runs (Nuclear Option)
```python
# Delete all runs in the experiment
experiment = client.get_experiment_by_name("EURUSD_Experiments")
runs = client.search_runs(experiment.experiment_id)
for run in runs:
    client.delete_run(run.info.run_id)
```

### After Deletion - Retrain:
```bash
# 1. Preprocess data
python src/02_preprocess.py

# 2. Train all models (with new artifact logging)
python src/03_train_models.py

# 3. Select champion
python src/04_evaluate_select.py

# 4. Restart API
# (on EC2 or locally)
```

---

## Quick Reference Commands

### Local Testing
```bash
# Run API locally
python api/app_cloud.py

# Test endpoints
curl http://localhost:8080/health
curl http://localhost:8080/api/predict
```

### EC2 Deployment
```bash
# Build and deploy
docker build -f Dockerfile.cloud -t eurusd-api .
docker run -d -p 8080:8080 \
  -e USE_S3=true \
  -e S3_BUCKET=eurusd-ml-models \
  -e MLFLOW_TRACKING_URI=http://localhost:5000 \
  eurusd-api

# Check logs
docker logs -f <container-id>
```

### Model Management
```bash
# Train new models
python src/03_train_models.py

# Select champion
python src/04_evaluate_select.py

# Restart API to load new champion
docker restart <container-id>
```
