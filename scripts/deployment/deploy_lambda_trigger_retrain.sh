#!/bin/bash
#
# Deploy Lambda Function to Trigger EC2 Retraining
#
# This script creates a Lambda function that starts the retraining EC2 instance
# on a daily schedule (EventBridge trigger).
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=========================================="
echo "Lambda Trigger Deployment"
echo -e "==========================================${NC}"

# Configuration
LAMBDA_NAME="eurusd-trigger-retrain"
LAMBDA_ROLE_NAME="eurusd-lambda-trigger-role"
LAMBDA_DIR="aws_lambda/trigger_retrain"
DEPLOYMENT_PACKAGE="/tmp/${LAMBDA_NAME}.zip"
AWS_REGION="${AWS_REGION:-us-east-1}"
RETRAIN_EC2_TAG="${RETRAIN_EC2_TAG:-eurusd-retrain-worker}"
SCHEDULE_EXPRESSION="cron(10 22 ? * MON-FRI *)"  # 22:10 UTC, weekdays only

echo -e "\n${YELLOW}Configuration:${NC}"
echo "  Lambda Name: $LAMBDA_NAME"
echo "  Region: $AWS_REGION"
echo "  EC2 Tag: $RETRAIN_EC2_TAG"
echo "  Schedule: $SCHEDULE_EXPRESSION (22:10 UTC, Mon-Fri)"

# Check if Lambda directory exists
if [ ! -d "$LAMBDA_DIR" ]; then
    echo -e "${RED}ERROR: Lambda directory not found: $LAMBDA_DIR${NC}"
    exit 1
fi

# 1. Create IAM Role for Lambda
echo -e "\n${GREEN}[1/5] Creating IAM Role for Lambda...${NC}"

# Check if role exists
if aws iam get-role --role-name $LAMBDA_ROLE_NAME 2>/dev/null; then
    echo "  IAM role $LAMBDA_ROLE_NAME already exists"
    ROLE_ARN=$(aws iam get-role --role-name $LAMBDA_ROLE_NAME --query "Role.Arn" --output text)
else
    echo "  Creating IAM role..."
    
    # Trust policy
    cat > /tmp/lambda-trust-policy.json <<EOF
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

    ROLE_ARN=$(aws iam create-role \
        --role-name $LAMBDA_ROLE_NAME \
        --assume-role-policy-document file:///tmp/lambda-trust-policy.json \
        --query "Role.Arn" --output text)
    
    # Attach basic Lambda execution policy
    aws iam attach-role-policy \
        --role-name $LAMBDA_ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    
    # EC2 start permissions
    cat > /tmp/lambda-ec2-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances",
        "ec2:StartInstances",
        "ec2:DescribeTags"
      ],
      "Resource": "*"
    }
  ]
}
EOF

    aws iam put-role-policy \
        --role-name $LAMBDA_ROLE_NAME \
        --policy-name EC2StartPolicy \
        --policy-document file:///tmp/lambda-ec2-policy.json
    
    echo "  ✅ IAM role created: $ROLE_ARN"
    
    # Wait for role to propagate
    echo "  Waiting for IAM role to propagate..."
    sleep 10
fi

# 2. Package Lambda Function
echo -e "\n${GREEN}[2/5] Packaging Lambda Function...${NC}"

# Clean up old package
rm -f $DEPLOYMENT_PACKAGE

# Create deployment package
cd $LAMBDA_DIR
zip -r $DEPLOYMENT_PACKAGE lambda_function_trigger.py
cd - > /dev/null

echo "  ✅ Package created: $DEPLOYMENT_PACKAGE"

# 3. Create/Update Lambda Function
echo -e "\n${GREEN}[3/5] Deploying Lambda Function...${NC}"

# Check if Lambda exists
if aws lambda get-function --function-name $LAMBDA_NAME 2>/dev/null; then
    echo "  Updating existing Lambda function..."
    
    aws lambda update-function-code \
        --function-name $LAMBDA_NAME \
        --zip-file fileb://$DEPLOYMENT_PACKAGE
    
    # Update configuration
    aws lambda update-function-configuration \
        --function-name $LAMBDA_NAME \
        --environment "Variables={RETRAIN_EC2_TAG_NAME=$RETRAIN_EC2_TAG}"
    
    echo "  ✅ Lambda function updated"
else
    echo "  Creating new Lambda function..."
    
    aws lambda create-function \
        --function-name $LAMBDA_NAME \
        --runtime python3.11 \
        --role $ROLE_ARN \
        --handler lambda_function_trigger.lambda_handler \
        --zip-file fileb://$DEPLOYMENT_PACKAGE \
        --timeout 60 \
        --memory-size 128 \
        --environment "Variables={RETRAIN_EC2_TAG_NAME=$RETRAIN_EC2_TAG}" \
        --description "Triggers EC2 instance for daily model retraining"
    
    echo "  ✅ Lambda function created"
fi

# Get Lambda ARN
LAMBDA_ARN=$(aws lambda get-function --function-name $LAMBDA_NAME --query "Configuration.FunctionArn" --output text)
echo "  Lambda ARN: $LAMBDA_ARN"

# 4. Create EventBridge Rule
echo -e "\n${GREEN}[4/5] Creating EventBridge Schedule...${NC}"

RULE_NAME="${LAMBDA_NAME}-schedule"

# Check if rule exists
if aws events describe-rule --name $RULE_NAME 2>/dev/null; then
    echo "  EventBridge rule already exists"
else
    echo "  Creating EventBridge rule..."
    
    aws events put-rule \
        --name $RULE_NAME \
        --schedule-expression "$SCHEDULE_EXPRESSION" \
        --state ENABLED \
        --description "Daily trigger for model retraining (weekdays at 22:10 UTC)"
    
    echo "  ✅ EventBridge rule created"
fi

# Get rule ARN
RULE_ARN=$(aws events describe-rule --name $RULE_NAME --query "Arn" --output text)

# 5. Add Lambda Permission for EventBridge
echo -e "\n${GREEN}[5/5] Configuring Permissions...${NC}"

# Remove existing permission if it exists
aws lambda remove-permission \
    --function-name $LAMBDA_NAME \
    --statement-id EventBridgeInvoke 2>/dev/null || true

# Add permission
aws lambda add-permission \
    --function-name $LAMBDA_NAME \
    --statement-id EventBridgeInvoke \
    --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn $RULE_ARN

echo "  ✅ Lambda permission added"

# Add Lambda as target to EventBridge rule
aws events put-targets \
    --rule $RULE_NAME \
    --targets "Id=1,Arn=$LAMBDA_ARN"

echo "  ✅ EventBridge target configured"

# Summary
echo -e "\n${GREEN}==========================================${NC}"
echo -e "${GREEN}Deployment Summary${NC}"
echo -e "${GREEN}==========================================${NC}"
echo "  ✅ Lambda Function: $LAMBDA_NAME"
echo "  ✅ Lambda ARN: $LAMBDA_ARN"
echo "  ✅ IAM Role: $LAMBDA_ROLE_NAME"
echo "  ✅ EventBridge Rule: $RULE_NAME"
echo "  ✅ Schedule: $SCHEDULE_EXPRESSION"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Test Lambda manually:"
echo "     aws lambda invoke --function-name $LAMBDA_NAME /tmp/response.json"
echo "     cat /tmp/response.json"
echo ""
echo "  2. Monitor CloudWatch Logs:"
echo "     aws logs tail /aws/lambda/$LAMBDA_NAME --follow"
echo ""
echo "  3. Verify EC2 starts:"
echo "     aws ec2 describe-instances --filters \"Name=tag:Name,Values=$RETRAIN_EC2_TAG\""
echo ""
echo "  4. Check EventBridge rule:"
echo "     aws events describe-rule --name $RULE_NAME"
echo ""
echo -e "${GREEN}Deployment complete!${NC}"
echo -e "${YELLOW}The Lambda will trigger daily at 22:10 UTC (Mon-Fri)${NC}"
