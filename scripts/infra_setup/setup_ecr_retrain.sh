#!/bin/bash
# setup_ecr_retrain.sh - Provision Amazon ECR for Model Retraining
#
# This script creates a private Amazon ECR repository and ensures the 
# retraining IAM role has the necessary permissions to pull images.

set -e

# Configuration
REPO_NAME="eurusd-retrain"
REGION="${AWS_REGION:-us-east-1}"
IAM_ROLE_NAME="eurusd-retrain-role"

echo "🐳 Amazon ECR Setup for EUR/USD Retraining"
echo "=========================================="

# 1. Create ECR Repository
echo "[1/2] Checking ECR Repository: $REPO_NAME..."
if aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$REGION" >/dev/null 2>&1; then
    echo "  ✅ Repository already exists."
else
    echo "  Creating repository..."
    aws ecr create-repository \
        --repository-name "$REPO_NAME" \
        --region "$REGION" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256
    echo "  ✅ Repository created successfully."
fi

# Get Repository URI
REPO_URI=$(aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$REGION" --query "repositories[0].repositoryUri" --output text)
echo "  📍 Repository URI: $REPO_URI"

# 2. Attach IAM Permissions
echo -e "\n[2/2] Configuring IAM Permissions for: $IAM_ROLE_NAME..."
if aws iam get-role --role-name "$IAM_ROLE_NAME" >/dev/null 2>&1; then
    echo "  Attaching AmazonEC2ContainerRegistryReadOnly policy..."
    aws iam attach-role-policy \
        --role-name "$IAM_ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
    echo "  ✅ IAM permissions updated."
else
    echo "  ⚠️  IAM role $IAM_ROLE_NAME not found (skipping policy attachment)."
    echo "     This role is usually created by scripts/deployment/deploy_retrain_ec2.sh"
fi

echo -e "\n=========================================="
echo "🚀 ECR Setup Complete!"
echo "   You can now build and push your image to:"
echo "   $REPO_URI"
echo "=========================================="
