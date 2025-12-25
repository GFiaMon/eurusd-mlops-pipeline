#!/bin/bash
# deploy.sh - Deployment script for AWS EC2

set -e

echo "🚀 EUR/USD ML App Deployment Script"
echo "===================================="

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory (two levels up from scripts/deployment)
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "📂 Directories:"
echo "  Script Dir: $SCRIPT_DIR"

# Smart Project Root Detection
SEARCH_DIR="$SCRIPT_DIR"
PROJECT_ROOT=""
for i in {1..4}; do
    if [ -f "$SEARCH_DIR/Dockerfile.cloud" ]; then
        PROJECT_ROOT="$SEARCH_DIR"
        break
    fi
    SEARCH_DIR="$(dirname "$SEARCH_DIR")"
done

if [ -z "$PROJECT_ROOT" ]; then
    echo "❌ ERROR: Could not find 'Dockerfile.cloud' in any parent directory."
    echo "   Current Directory: $(pwd)"
    echo "   Script Directory: $SCRIPT_DIR"
    echo "   Contents of current dir:"
    ls -F
    exit 1
fi

echo "  Project Root (Detected): $PROJECT_ROOT"
echo ""

# Configuration
APP_NAME="eurusd-predictor"
DOCKER_IMAGE="eurusd-predictor:latest"
CONTAINER_NAME="eurusd-app"
PORT=8080

# Parse command line arguments
STORAGE_TYPE=${1:-"s3"}
S3_BUCKET=${2:-"eurusd-ml-models"}
MLFLOW_URI=${3:-"http://localhost:5000"} 

# ... (Host connection fix part remains the same) ...
# [Keep lines 31-63 as they were in previous version]

# Stop and remove existing container
echo "🛑 Stopping existing container..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Build Docker image from project root using Dockerfile.cloud
echo "🔨 Building Docker image..."
cd "$PROJECT_ROOT"
echo "Deploying Flask API to EC2..."
echo "Config: KEY_PEM=$KEY_PEM, EC2_IP=$EC2_IP"

# 0. Sync DataManager to API directory (Deployment Copy)
echo "📋 Syncing DataManager to API..."
mkdir -p api/utils
# Only copy if source exists (when running from project root)
if [ -f "utils/data_manager.py" ]; then
    cp utils/data_manager.py api/utils/data_manager.py
    echo "✅ DataManager copied from utils/"
elif [ -f "api/utils/data_manager.py" ]; then
    echo "✅ DataManager already in api/utils/"
else
    echo "❌ ERROR: data_manager.py not found in utils/ or api/utils/"
    exit 1
fi
# Create __init__.py if it doesn't exist
touch api/utils/__init__.py
echo "✅ DataManager synced"

# 1. Prepare Docker files (ensure they are in api/)me
echo "   📍 Currently in: $(pwd)"

# Check for Dockerfile one last time
if [ ! -f "Dockerfile.cloud" ]; then
    echo "   ❌ CRITICAL: Dockerfile.cloud is missing in $(pwd)!"
    ls -la
    exit 1
fi

# Ensure 'data' directory exists so Docker COPY doesn't fail
if [ ! -d "data" ]; then
    echo "  📁 Creating empty 'data' directory for build context..."
    mkdir -p data/raw data/processed
fi

echo "   🚀 Running docker build..."
docker build -f "$PROJECT_ROOT/Dockerfile.cloud" -t $DOCKER_IMAGE .

# Run container based on storage type
echo "🏃 Starting container..."

if [ "$STORAGE_TYPE" = "s3" ]; then
    echo "  Using S3 and MLflow..."
    docker run -d \
        --name $CONTAINER_NAME \
        -p $PORT:$PORT \
        -e USE_S3=true \
        -e S3_BUCKET=$S3_BUCKET \
        -e MLFLOW_TRACKING_URI=$MLFLOW_URI \
        -e AWS_REGION=us-east-1 \
        -e PORT=$PORT \
        --restart unless-stopped \
        $DOCKER_IMAGE
else
    echo "  Using local storage..."
    docker run -d \
        --name $CONTAINER_NAME \
        -p $PORT:$PORT \
        -e USE_S3=false \
        -e MLFLOW_TRACKING_URI=$MLFLOW_URI \
        -e PORT=$PORT \
        --restart unless-stopped \
        $DOCKER_IMAGE
fi

# Wait for container to start
echo "⏳ Waiting for container to start..."
sleep 5

# Check if container is running
if docker ps | grep -q $CONTAINER_NAME; then
    echo "✅ Container started successfully!"
    echo ""
    echo "📊 Container status:"
    docker ps | grep $CONTAINER_NAME
    echo ""
    echo "📝 Recent logs:"
    docker logs --tail 20 $CONTAINER_NAME
    echo ""
    echo "🌐 Application should be available at:"
    echo "   http://localhost:$PORT (if running locally)"
    echo "   http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo 'YOUR-EC2-IP'):$PORT (if on EC2)"
    echo ""
    echo "💡 Useful commands:"
    echo "   View logs: docker logs -f $CONTAINER_NAME"
    echo "   Stop app: docker stop $CONTAINER_NAME"
    echo "   Restart app: docker restart $CONTAINER_NAME"
    echo "   Check health: curl http://localhost:$PORT/health"
    
    # Optional: Clean up dangling images to save space
    echo ""
    echo "🧹 Cleaning up old unused images..."
    docker image prune -f
else
    echo "❌ Container failed to start!"
    echo "📝 Logs:"
    docker logs $CONTAINER_NAME
    exit 1
fi
