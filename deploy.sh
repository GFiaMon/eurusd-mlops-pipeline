#!/bin/bash
# deploy.sh - Deployment script for AWS EC2

set -e

echo "🚀 EUR/USD ML App Deployment Script"
echo "===================================="

# Configuration
APP_NAME="eurusd-predictor"
DOCKER_IMAGE="eurusd-predictor:latest"
CONTAINER_NAME="eurusd-app"
PORT=8080

# Parse command line arguments
STORAGE_TYPE=${1:-"local"}  # local or s3
S3_BUCKET=${2:-"eurusd-ml-models"}

echo ""
echo "📋 Configuration:"
echo "  Storage Type: $STORAGE_TYPE"
echo "  Docker Image: $DOCKER_IMAGE"
echo "  Container Name: $CONTAINER_NAME"
echo "  Port: $PORT"

if [ "$STORAGE_TYPE" = "s3" ]; then
    echo "  S3 Bucket: $S3_BUCKET"
fi

echo ""

# Stop and remove existing container
echo "🛑 Stopping existing container..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Build Docker image
echo "🔨 Building Docker image..."
docker build -t $DOCKER_IMAGE .

# Run container based on storage type
echo "🏃 Starting container..."

if [ "$STORAGE_TYPE" = "s3" ]; then
    echo "  Using S3 storage..."
    docker run -d \
        --name $CONTAINER_NAME \
        -p 80:$PORT \
        -e USE_S3=true \
        -e S3_BUCKET=$S3_BUCKET \
        -e S3_MODEL_KEY=models/lstm_trained_model.keras \
        -e S3_SCALER_KEY=models/lstm_scaler.joblib \
        -e S3_DATA_KEY=data/processed/lstm_simple_test_data.csv \
        -e AWS_REGION=us-east-1 \
        -e PORT=$PORT \
        --restart unless-stopped \
        $DOCKER_IMAGE
else
    echo "  Using local storage..."
    docker run -d \
        --name $CONTAINER_NAME \
        -p 80:$PORT \
        -e USE_S3=false \
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
    echo "   http://localhost (if running locally)"
    echo "   http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo 'YOUR-EC2-IP') (if on EC2)"
    echo ""
    echo "💡 Useful commands:"
    echo "   View logs: docker logs -f $CONTAINER_NAME"
    echo "   Stop app: docker stop $CONTAINER_NAME"
    echo "   Restart app: docker restart $CONTAINER_NAME"
    echo "   Check health: curl http://localhost/health"
else
    echo "❌ Container failed to start!"
    echo "📝 Logs:"
    docker logs $CONTAINER_NAME
    exit 1
fi
