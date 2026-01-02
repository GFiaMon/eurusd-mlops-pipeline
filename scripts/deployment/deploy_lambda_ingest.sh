#!/bin/bash
set -e

# Configuration
FUNCTION_NAME="eurusd-ingest"
ROLE_NAME="EurusdIngestLambdaRole"
POLICY_NAME="EurusdIngestS3Policy"
RULE_NAME="eurusd-ingest-daily"
REGION="us-east-1" # Change if needed or make dynamic
# Load from .env if available for bucket/prefix
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Fallback/Default
S3_BUCKET=${S3_BUCKET:-"eurusd-ml-models"} 
S3_PREFIX=${S3_PREFIX:-"data/raw/"}

echo "Deploying $FUNCTION_NAME to $REGION..."
echo "Config: S3_BUCKET=$S3_BUCKET, S3_PREFIX=$S3_PREFIX"

# 1. IAM Role & Policy
echo "Creating IAM Role..."
cat > trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

aws iam create-role --role-name $ROLE_NAME --assume-role-policy-document file://trust-policy.json || echo "Role exists"

# S3 Policy
cat > role-policy.json <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        },
        {
            "Effect": "Allow",
            "Action": "s3:ListBucket",
            "Resource": "arn:aws:s3:::$S3_BUCKET"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject"
            ],
            "Resource": "arn:aws:s3:::$S3_BUCKET/$S3_PREFIX*"
        }
    ]
}
EOF

aws iam put-role-policy --role-name $ROLE_NAME --policy-name $POLICY_NAME --policy-document file://role-policy.json
echo "IAM Role configured."

# Wait for role propagation
echo "Waiting for role propagation..."
sleep 10

# 2. Package Lambda using Docker (to assume Linux environment and build binaries)
echo "Packaging Lambda using Docker..."
rm -rf build_package deployment_package.zip
mkdir -p build_package

# 0. Sync DataManager to Lambda directory (Deployment Copy)
echo "📋 Syncing DataManager to Lambda package..."
# Copy to source dir first so it's included in zip later if needed, 
# OR copy directly to build package. 
# Better strategy: Copy to source dir (gitignored) so structure is valid locally too.
cp utils/data_manager.py aws_lambda/data_ingestion/data_manager.py
echo "✅ DataManager synced"

# Define local path for mounting
LOCAL_PATH=$(pwd)/build_package

# Use Docker to install dependencies compatible with AWS Lambda (Python 3.11)
# We use the official AWS SAM build image for maximum compatibility
echo "Installing dependencies via Docker (public.ecr.aws/sam/build-python3.11)..."
docker run --platform linux/amd64 --rm -v "$LOCAL_PATH":/var/task public.ecr.aws/sam/build-python3.11 \
    /bin/sh -c "pip install \"yfinance==0.2.66\" \"peewee==3.18.3\" \"pandas==2.1.4\" \"numpy==1.24.3\" \"lxml==6.0.2\" \"requests==2.32.5\" -t /var/task --upgrade"

# Copy function code (after build to avoid it being overwritten or needing mount)
cp aws_lambda/data_ingestion/lambda_function_ingest.py build_package/
# Copy DataManager into build package
cp aws_lambda/data_ingestion/data_manager.py build_package/

# Zip
cd build_package
zip -r ../deployment_package.zip .
cd ..

# 3. Deploy Function
ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text)
DEPLOY_KEY="deployment/eurusd_ingest.zip"

echo "Uploading package to s3://$S3_BUCKET/$DEPLOY_KEY..."
aws s3 cp deployment_package.zip s3://$S3_BUCKET/$DEPLOY_KEY

if aws lambda get-function --function-name $FUNCTION_NAME >/dev/null 2>&1; then
    echo "Updating function code..."
    aws lambda update-function-code --function-name $FUNCTION_NAME --s3-bucket $S3_BUCKET --s3-key $DEPLOY_KEY
    
    echo "Waiting for update to complete..."
    aws lambda wait function-updated --function-name $FUNCTION_NAME
    
    echo "Updating configuration..."
    aws lambda update-function-configuration --function-name $FUNCTION_NAME \
        --runtime python3.11 \
        --environment "Variables={S3_BUCKET=$S3_BUCKET,S3_PREFIX=$S3_PREFIX}"
else
    echo "Creating function..."
    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --code S3Bucket=$S3_BUCKET,S3Key=$DEPLOY_KEY \
        --handler lambda_function_ingest.lambda_handler \
        --runtime python3.11 \
        --role $ROLE_ARN \
        --timeout 300 \
        --memory-size 512 \
        --environment "Variables={S3_BUCKET=$S3_BUCKET,S3_PREFIX=$S3_PREFIX}"
fi

# 4. Schedule (EventBridge)
echo "Setting up Schedule..."
# 5:00 PM EST = 22:00 UTC (Standard Time)
# Weekdays only (Mon-Fri)
aws events put-rule --name $RULE_NAME --schedule-expression "cron(0 22 ? * MON-FRI *)"

aws lambda add-permission \
    --function-name $FUNCTION_NAME \
    --statement-id "EventBridgeInvoke" \
    --action "lambda:InvokeFunction" \
    --principal events.amazonaws.com \
    --source-arn $(aws events describe-rule --name $RULE_NAME --query 'Arn' --output text) \
    || echo "Permission already exists"

aws events put-targets --rule $RULE_NAME --targets "Id"="1","Arn"=$(aws lambda get-function --function-name $FUNCTION_NAME --query 'Configuration.FunctionArn' --output text)

# Cleanup
rm trust-policy.json role-policy.json deployment_package.zip
rm -rf build_package

echo "Deployment Complete! 🚀"
