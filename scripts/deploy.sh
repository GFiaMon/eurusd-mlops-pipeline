#!/bin/bash
# deploy.sh - Deployment script for AWS EC2

set -e

echo "🚀 EUR/USD ML App Deployment Script"
echo "===================================="

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory (one level up from scripts)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo ""
echo "📂 Directories:"
echo "  Script Dir: $SCRIPT_DIR"
echo "  Project Root: $PROJECT_ROOT"
echo ""

# Configuration
APP_NAME="eurusd-predictor"
DOCKER_IMAGE="eurusd-predictor:latest"
CONTAINER_NAME="eurusd-app"
PORT=8080

# Parse command line arguments
STORAGE_TYPE=${1:-"s3"}  # default to s3 for cloud deployment
S3_BUCKET=${2:-"eurusd-ml-models"}
MLFLOW_URI=${3:-"http://localhost:5000"} 

# 🚨 HOST CONNECTION FIX: Inside Docker, 'localhost' refers to the container.
# If the user passed localhost, we try to detect the EC2 Public IP to allow the container to reach the host.
if [[ "$MLFLOW_URI" == *"localhost"* ]]; then
    echo "⚠️ Detected 'localhost' in MLflow URI. Inside Docker, this refers to the container, not the host."
    echo "   Attempting to detect EC2 Public IP..."
    PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "")
    if [ ! -z "$PUBLIC_IP" ]; then
        MLFLOW_URI=${MLFLOW_URI/localhost/$PUBLIC_IP}
        echo "   🔄 Updated MLflow URI to: $MLFLOW_URI"
    else
        echo "   ❌ Could not detect Public IP. You may need to provide it manually."
    fi
fi

echo ""
echo "📋 Configuration:"
echo "  Storage Type: $STORAGE_TYPE"
echo "  Docker Image: $DOCKER_IMAGE"
echo "  Container Name: $CONTAINER_NAME"
echo "  Port: $PORT"
echo "  MLflow URI: $MLFLOW_URI"

if [ "$STORAGE_TYPE" = "s3" ]; then
    echo "  S3 Bucket: $S3_BUCKET"
fi

echo ""

# Stop and remove existing container
echo "🛑 Stopping existing container..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Build Docker image from project root using Dockerfile.cloud
echo "🔨 Building Docker image using Dockerfile.cloud..."
cd "$PROJECT_ROOT"
docker build -f Dockerfile.cloud -t $DOCKER_IMAGE .

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
