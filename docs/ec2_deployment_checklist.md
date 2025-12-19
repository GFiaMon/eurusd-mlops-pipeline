# EC2 Deployment Checklist - Flask API Update

## Prerequisites
- [x] MLflow server running on EC2 (http://44.220.55.185:5000)
- [x] S3 bucket configured (eurusd-ml-models)
- [x] Models trained and champion selected
- [ ] EC2 instance has at least 2GB RAM (t2.small minimum)
- [ ] **AWS Security Group**: Ports **8080** and **5000** open to your IP

## Step 1: Clean Up EC2 (Recommended)

### 1. Delete macOS junk files
Run this on your EC2 home directory to clean up:
```bash
find ~ -name "._*" -delete
```

### 2. Recommended Structure (Root)
Since you prefer to keep everything in the home directory, your folder structure should look like this:
```text
/home/ec2-user/
├── api/              # Your Flask app code (REQUIRED)
├── scripts/          # Deployment scripts (REQUIRED)
├── data/             # Data for Docker build context (REQUIRED)
├── Dockerfile.cloud   # Build instructions (REQUIRED)
└── .env              # Environment variables
```

### 3. Safe to Delete (Obsolete Files 🧹)

> [!TIP]
> **Keep the `data/` folder**: Even though the app uses S3, the `Dockerfile.cloud` needs the `data/` folder during the **build process** to create the image.
>
> **Delete the `models/` folder**: This is safe to delete because the app fetches the model directly from MLflow/S3 at runtime—it never looks at the host's `models/` folder.

**Run this command to remove the truly obsolete items:**
```bash
# Safely remove only legacy/unused files
rm -rf models eurusd-app.tar.gz Dockerfile deploy.sh Dockerfile.api
```
rm -rf models eurusd-app.tar.gz Dockerfile deploy.sh Dockerfile.api


## Step 2: Update Local Dependencies

```bash
# Update requirements
pip install -r api/requirements.txt

# Test locally first
python api/app_cloud.py
```

## Step 3: Deploy to EC2

### 1. Transfer Code
Run these from your **local terminal** (where your project is):

**Method A: SCP via SSH Alias (Simplest)**
If you have your EC2 set up in `~/.ssh/config` as `capst2-inference-app`:
```bash
# Upload updated folders
scp -r api/ capst2-inference-app:~/
scp scripts/deploy.sh capst2-inference-app:~/scripts/

# Upload the Dockerfile
scp Dockerfile.cloud capst2-inference-app:~/

# Note: If the scripts folder doesn't exist on EC2, run 'mkdir ~/scripts' first
```

**Method B: Standard SCP (If no SSH config)**
```bash
# Upload to home directory
scp -i eurusd-keypair.pem -r api/ ec2-user@<ec2-ip>:~/
scp -i eurusd-keypair.pem scripts/deploy.sh ec2-user@<ec2-ip>:~/scripts/
scp -i eurusd-keypair.pem Dockerfile.cloud ec2-user@<ec2-ip>:~/
```

**Method C: Git**
```bash
# Just run pull in root
git pull origin main
```

### 2. Connect and Run
Now log into your EC2:
```bash
ssh capst2-inference-app
# You are now in the home directory
```

### 3. Run your app (Option A: Automated Script 🚀)
Use the script to handle build, stop, and run in one go:
```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh s3 eurusd-ml-models http://localhost:5000
```
> [!NOTE]
> The updated `deploy.sh` will automatically replace `localhost` with your EC2 Public IP so the container can connect to MLflow on the host.

This is the easiest way as it handles all the environment variables, port mappings, and **automatically cleans up old images** to save disk space!

### 4. Run your app (Option B: Manual Commands)
Use this if you want full control or to troubleshoot specific steps:
```bash
# 1. Build
docker build -f Dockerfile.cloud -t eurusd-api:latest .

# 2. Stop old container
docker stop eurusd-app 2>/dev/null || true
docker rm eurusd-app 2>/dev/null || true

# 3. Run
docker run -d \
  --name eurusd-app \
  --restart unless-stopped \
  -p 8080:8080 \
  -e USE_S3=true \
  -e S3_BUCKET=eurusd-ml-models \
  -e MLFLOW_TRACKING_URI=http://localhost:5000 \
  -e AWS_REGION=us-east-1 \
  eurusd-api:latest
```

> [!TIP]
> **Good Practice**: After a successful manual deploy, run `docker image prune -f` to delete the previous version of the image and free up space.

### Verify Deployment
```bash
# Check container is running
docker ps

# Check logs
docker logs -f eurusd-app

# Test health endpoint
curl http://localhost:8080/health

# Test prediction
curl http://localhost:8080/api/predict
```

## Step 4: Test from Outside EC2

```bash
# From your local machine
curl http://<ec2-public-ip>:8080/health
curl http://<ec2-public-ip>:8080/api/predict
```

## Troubleshooting

### Container Won't Start / No Space Left ⚠️
If you see **"Errno 28: No space left on device"**, it means Docker's cache and old images have filled up your EC2 disk.

**1. Check your disk space:**
```bash
df -h
```

**2. Immediate Fix (Clear Docker Cache):**
If you are at 100% usage, run the "Nuclear" cleanup below! It's the only way to free enough space for heavy libraries like TensorFlow.

```bash
docker system prune -a --volumes -f
```

**3. Common issues:**
- MLflow connection: Check MLFLOW_TRACKING_URI
- S3 permissions: Verify EC2 IAM role has S3 access
- **Disk Full**: Run `docker system prune -a --volumes -f`

### Connection Trap: "Localhost" inside Docker 🐳
If you see `Connection refused` in `docker logs eurusd-app`:
- Inside Docker, `localhost` means the container itself, not the EC2.
- The `deploy.sh` script now tries to fix this automatically.
- If it fails, run the script with your public IP manually:
  ```bash
  ./scripts/deploy.sh s3 eurusd-ml-models http://44.220.55.185:5000
  ```

### Model Not Loading
```bash
# Verify MLflow server is accessible
curl http://localhost:5000/health

# Check S3 bucket
aws s3 ls s3://eurusd-ml-models/models/

# Verify best_model_info.json exists
aws s3 cp s3://eurusd-ml-models/models/best_model_info.json -
```

### 500 Errors
```bash
# Check container logs for Python errors
docker logs eurusd-app --tail 100

# Common causes:
# - Data type issues (check feature engineering)
# - Shape mismatches (verify LSTM input shape)
# - Missing scaler (retrain models with artifact logging)
```

### Can't Connect from Your Computer
If you get "Failed to connect" from your local terminal, check your **AWS Security Group**:

1. Go to AWS Console > EC2 > Instances.
2. Select your instance > Security tab > Click the Security Group ID.
3. **Inbound Rules**: Add 2 rules:
   - **Type**: Custom TCP, **Port**: 8080, **Source**: My IP
   - **Type**: Custom TCP, **Port**: 5000, **Source**: My IP (for MLflow)

## Updating After New Model Training

When you train a new champion model:

```bash
# 1. Train locally
python src/03_train_models.py
python src/04_evaluate_select.py

# 2. SSH to EC2
ssh capst2-inference-app

# 3. Just restart the container (no rebuild needed!)
docker restart eurusd-app

# 4. Verify new model loaded
docker logs eurusd-app | grep "Model loaded"
```

## Environment Variables Reference

| Variable | Value | Purpose |
|----------|-------|---------|
| `USE_S3` | `true` | Enable S3 for model info |
| `S3_BUCKET` | `eurusd-ml-models` | Your S3 bucket name |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `AWS_REGION` | `us-east-1` | AWS region |
| `PORT` | `8080` | API port (optional) |

## Quick Commands

```bash
# Restart API
docker restart eurusd-app

# View logs
docker logs -f eurusd-app

# Stop API
docker stop eurusd-app

# Remove container
docker rm eurusd-app

# Rebuild image
docker build -f Dockerfile.cloud -t eurusd-api:latest .
```

## 🛠️ Docker Maintenance & Cleanup

Small EC2 instances (like `t2.small`) can run out of disk space quickly because Docker saves every old version of your images. 

### 1. Recommended Maintenance (Safe ✅)
Run this periodically to delete old, unused images and free up space:
```bash
docker image prune -f
```

### 2. The "Nuclear" Option (Use with extreme caution! ⚠️)
The commands you found are great for a total reset, but **be careful**: 

> [!CAUTION]
> **DANGER**: If your **MLflow server** or **Database** is running as a Docker container on this same EC2, the commands below will stop and delete them too!
> 
> - `docker stop $(docker ps -q)` will stop **all** running containers.
> - `docker system prune -a --volumes` will delete **all** stopped containers and **all** volumes (potentially deleting your MLflow database if it's not backed up).

If you want to wipe **everything** Docker-related:
```bash
# Stop all containers (MLflow, DB, everything!)
docker stop $(docker ps -q)

# Remove all containers, volumes, and images
docker system prune -a --volumes -f
```

**Why we don't use this in our script?**
In `deploy.sh`, we only use `docker stop eurusd-app` because we want to be safe and only update the API without affecting your other ML infrastructure.
