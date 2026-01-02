#!/bin/bash
#
# EC2 User Data Script for Model Retraining
#
# This script sets up a persistent "per-boot" script that:
# 1. Runs automatically on every instance start (boot).
# 2. Handles Docker installation/updates.
# 3. Runs the retraining container.
# 4. Shuts down the instance upon completion.
#

set -e

# ==============================================================================
# 1. Create Configuration File
# ==============================================================================
cat <<EOF > /etc/eurusd-retrain.conf
# Configuration for EUR/USD Retraining
S3_BUCKET="${S3_BUCKET:-eurusd-ml-models}"
AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPOSITORY_URI="${ECR_REPOSITORY_URI:-238708039523.dkr.ecr.us-east-1.amazonaws.com/eurusd-retrain}"
LOG_FILE="/var/log/retrain.log"
EOF

# ==============================================================================
# 2. Create the Retraining Script
# ==============================================================================
cat <<'SCRIPT_EOF' > /usr/local/bin/eurusd-retrain.sh
#!/bin/bash
# This script is executed on every boot via cloud-init per-boot configuration.

# Load Configuration
source /etc/eurusd-retrain.conf

# Setup Logging
# Redirect stdout and stderr to the log file, but also keep sending to console (fd 1)
# we use process substitution to append to LOG_FILE
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "EC2 Retraining Task Started"
echo "Time: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo "=========================================="

# ------------------------------------------------------------------
# Phase 1: Infrastructure Checks (Networking & Package Manager)
# ------------------------------------------------------------------
echo "[1/6] Configuring networking and checking package manager..."
echo "ip_resolve=4" >> /etc/dnf/dnf.conf 2>/dev/null || true

# Wait for background updates/locks (Critical Loop)
for i in {1..60}; do
    if ! fuser /var/lib/dnf/lock >/dev/null 2>&1 && ! fuser /var/cache/dnf >/dev/null 2>&1; then
        echo "  Package manager is free."
        break
    fi
    echo "  Waiting for background package manager processes ($i/60)..."
    sleep 5
done

# ------------------------------------------------------------------
# Phase 2: Ensure Docker is Installed & Running
# ------------------------------------------------------------------
if ! command -v docker &> /dev/null; then
    echo "[2/6] Installing Docker..."
    yum update -y
    yum install -y docker
    systemctl enable docker
    systemctl start docker
    usermod -a -G docker ec2-user
else
    echo "[2/6] Docker already installed. Ensuring service is running..."
    systemctl start docker
fi

# ------------------------------------------------------------------
# Phase 3: Authenticate & Pull Image
# ------------------------------------------------------------------
echo "[3/6] Authenticating to ECR and pulling image..."
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REPOSITORY_URI"
docker pull "$ECR_REPOSITORY_URI:latest"

# ------------------------------------------------------------------
# Phase 4: Run Retraining Container
# ------------------------------------------------------------------
echo "[4/6] Running retraining container..."

# Download .env if needed (optional, just in case)
aws s3 cp "s3://$S3_BUCKET/deployment/.env" ./ .env || true

# Run container with ERROR HANDLING (set +e equivalent logic)
# We capture exit code explicitly
set +e
docker run --name retrain-worker \
    -e S3_BUCKET="$S3_BUCKET" \
    -e AWS_REGION="$AWS_REGION" \
    -e RETRAIN_TOP_N=3 \
    -e MLFLOW_EC2_TAG_NAME="MLflow-Server" \
    -e RETRAIN_EC2_TAG_NAME="eurusd-retrain-worker" \
    "$ECR_REPOSITORY_URI:latest"

RETRAIN_EXIT_CODE=$?
set -e

if [ $RETRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Retraining container finished successfully."
else
    echo "❌ Retraining container FAILED with exit code $RETRAIN_EXIT_CODE."
fi

# ------------------------------------------------------------------
# Phase 5: Upload Logs
# ------------------------------------------------------------------
echo "[5/6] Uploading logs to S3..."
LOG_DATE=$(date -u +"%Y%m%d_%H%M%S")
S3_LOG_PATH="s3://$S3_BUCKET/logs/retrain_${LOG_DATE}.log"

# Capture container logs specific to the application
docker logs retrain-worker >> "$LOG_FILE" 2>&1 || echo "No container logs found."

# Upload the master log file
aws s3 cp "$LOG_FILE" "$S3_LOG_PATH" || echo "⚠️ Failed to upload logs to S3"

# Cleanup container
docker rm retrain-worker || true

# ------------------------------------------------------------------
# Phase 6: Shutdown
# ------------------------------------------------------------------
echo "=========================================="
echo "Shutting down instance..."
echo "Time: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo "=========================================="

sync
sleep 5
sudo poweroff
SCRIPT_EOF

chmod +x /usr/local/bin/eurusd-retrain.sh

# ==============================================================================
# 3. Register as Per-Boot Script
# ==============================================================================
# Linking to /var/lib/cloud/scripts/per-boot/ ensures it runs on every boot
mkdir -p /var/lib/cloud/scripts/per-boot
ln -sf /usr/local/bin/eurusd-retrain.sh /var/lib/cloud/scripts/per-boot/01_eurusd_retrain.sh

# ==============================================================================
# 4. Execute Immediately (for the first run)
# ==============================================================================
echo "Executing retraining script for the first time..."
/usr/local/bin/eurusd-retrain.sh
