#!/bin/bash

# To run this script:
# ./update_ec2_flask.sh
# ./update_ec2_flask.sh --restart-only (Quick Restart Mode)
# ./update_ec2_flask.sh --shell     (Interactive Shell Mode)


# Configuration
HOST="eurusd-inference-app"
REMOTE_DIR="~/eurusd-app"
MLFLOW_URI="http://34.229.131.42:5000"

# Parse arguments
RESTART_ONLY=false
OPEN_SHELL=false

for arg in "$@"
do
    case $arg in
        --restart-only)
        RESTART_ONLY=true
        shift
        ;;
        --shell)
        OPEN_SHELL=true
        shift
        ;;
    esac
done

# --- RESTART ONLY MODE ---
if [ "$RESTART_ONLY" = true ]; then
    echo "🔄  Restarting eurusd-app container on $HOST..."
    ssh $HOST "docker restart eurusd-app"
    echo "✅  Restart complete."
    exit 0
fi

# --- FULL UPDATE MODE ---

# 1. Local Cleanup
echo "🧹  Cleaning up local junk files..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name ".DS_Store" -delete

# 2. Upload Files
echo "🚀  Uploading code to $HOST..."
# Ensure remote scripts directory exists (just in case)
ssh $HOST "mkdir -p $REMOTE_DIR/scripts"

scp -r api $HOST:$REMOTE_DIR/
scp -r utils $HOST:$REMOTE_DIR/
scp scripts/deployment/deploy_flask_api.sh $HOST:$REMOTE_DIR/scripts/
scp Dockerfile.cloud $HOST:$REMOTE_DIR/

# 3. Remote Execution
echo "💻  Connecting to $HOST to run deployment..."
REMOTE_CMD="cd $REMOTE_DIR && chmod +x scripts/deploy_flask_api.sh && ./scripts/deploy_flask_api.sh s3 eurusd-ml-models $MLFLOW_URI"

if [ "$OPEN_SHELL" = true ]; then
    echo "ℹ️   Shell mode active: will remain logged in after deployment."
    # -t forces pseudo-tty allocation, allowing interactive shell
    ssh -t $HOST "$REMOTE_CMD; echo 'Deployment finished. Bash session started.'; exec bash -l"
else
    ssh -t $HOST "$REMOTE_CMD"
fi
