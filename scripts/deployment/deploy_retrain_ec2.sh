#!/bin/bash
#
# Deploy EC2 Instance for Model Retraining
#
# This script creates a dedicated EC2 instance that:
# - Starts on-demand via Lambda trigger
# - Runs the retraining script automatically
# - Shuts down after completion
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=========================================="
echo "EC2 Retraining Instance Deployment"
echo -e "==========================================${NC}"

# Configuration
INSTANCE_NAME="eurusd-retrain-worker"
INSTANCE_TYPE="${INSTANCE_TYPE:-t4g.small}"  # Use ARM64 Free Tier (2GB RAM)
# Automatically resolve the latest Amazon Linux 2023 ARM64 AMI
AMI_ID="${AMI_ID:-resolve:ssm:/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64}"
KEY_NAME="${KEY_NAME:-eurusd-keypair}"  # Using existing key pair
ECR_REPOSITORY_URI="238708039523.dkr.ecr.us-east-1.amazonaws.com/eurusd-retrain"
SECURITY_GROUP_NAME="eurusd-retrain-sg"
IAM_ROLE_NAME="eurusd-retrain-role"
S3_BUCKET="${S3_BUCKET:-eurusd-ml-models}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# User Data script path
USER_DATA_SCRIPT="scripts/infra_setup/user_data_retrain.sh"
PACKAGE_NAME="eurusd-retrain-code.tar.gz"
S3_PACKAGE_KEY="deployment/$PACKAGE_NAME"

echo -e "\n${YELLOW}Configuration:${NC}"
echo "  Instance Name: $INSTANCE_NAME"
echo "  Instance Type: $INSTANCE_TYPE"
echo "  AMI ID: $AMI_ID"
echo "  Key Pair: $KEY_NAME"
echo "  Region: $AWS_REGION"
echo "  S3 Bucket: $S3_BUCKET"

# Check if user data script exists
if [ ! -f "$USER_DATA_SCRIPT" ]; then
    echo -e "${RED}ERROR: User data script not found: $USER_DATA_SCRIPT${NC}"
    exit 1
fi

# 0. Package and Upload Code to S3
echo -e "\n${GREEN}[0/5] Packaging and Uploading Code to S3...${NC}"

TEMP_DIR="/tmp/eurusd-package-$(date +%s)"
rm -rf $TEMP_DIR
mkdir -p $TEMP_DIR/eurusd-capstone

# Copy necessary files
cp -r scripts $TEMP_DIR/eurusd-capstone/
cp -r utils $TEMP_DIR/eurusd-capstone/
cp requirements.txt $TEMP_DIR/eurusd-capstone/
cp .env.example $TEMP_DIR/eurusd-capstone/.env

# Create tarball
cd $TEMP_DIR
tar -czf $PACKAGE_NAME eurusd-capstone/
cd - > /dev/null

# Upload to S3
echo "  Uploading $PACKAGE_NAME to s3://$S3_BUCKET/$S3_PACKAGE_KEY..."
aws s3 cp $TEMP_DIR/$PACKAGE_NAME s3://$S3_BUCKET/$S3_PACKAGE_KEY

# Cleanup
rm -rf $TEMP_DIR

echo "  ✅ Code package synchronized to S3"

# 1. Create IAM Role
echo -e "\n${GREEN}[1/5] Creating IAM Role...${NC}"

# Check if role exists
if aws iam get-role --role-name $IAM_ROLE_NAME 2>/dev/null; then
    echo "  IAM role $IAM_ROLE_NAME already exists"
else
    echo "  Creating IAM role..."
    
    # Trust policy
    cat > /tmp/trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    aws iam create-role \
        --role-name $IAM_ROLE_NAME \
        --assume-role-policy-document file:///tmp/trust-policy.json
    
    # Attach policies
    echo "  Attaching policies..."
    
    # S3 access
    cat > /tmp/s3-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::${S3_BUCKET}",
        "arn:aws:s3:::${S3_BUCKET}/*"
      ]
    }
  ]
}
EOF

    aws iam put-role-policy \
        --role-name $IAM_ROLE_NAME \
        --policy-name S3Access \
        --policy-document file:///tmp/s3-policy.json
    
    # EC2 describe (to find MLflow server)
    cat > /tmp/ec2-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances",
        "ec2:DescribeTags"
      ],
      "Resource": "*"
    }
  ]
}
EOF

    aws iam put-role-policy \
        --role-name $IAM_ROLE_NAME \
        --policy-name EC2Describe \
        --policy-document file:///tmp/ec2-policy.json
    
    # CloudWatch Logs
    aws iam attach-role-policy \
        --role-name $IAM_ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess
    
    echo "  ✅ IAM role created"
fi

# Create instance profile
echo "  Creating instance profile..."
if aws iam get-instance-profile --instance-profile-name $IAM_ROLE_NAME 2>/dev/null; then
    echo "  Instance profile already exists"
else
    aws iam create-instance-profile --instance-profile-name $IAM_ROLE_NAME
    aws iam add-role-to-instance-profile \
        --instance-profile-name $IAM_ROLE_NAME \
        --role-name $IAM_ROLE_NAME
    echo "  ✅ Instance profile created"
    
    # Wait for instance profile to be ready
    echo "  Waiting for instance profile to propagate..."
    sleep 10
fi

# 2. Create Security Group
echo -e "\n${GREEN}[2/5] Creating Security Group...${NC}"

# Get default VPC
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text)

if [ "$VPC_ID" == "None" ] || [ -z "$VPC_ID" ]; then
    echo -e "${RED}ERROR: No default VPC found${NC}"
    exit 1
fi

echo "  Using VPC: $VPC_ID"

# Check if security group exists
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --query "SecurityGroups[0].GroupId" --output text 2>/dev/null)

if [ "$SG_ID" != "None" ] && [ -n "$SG_ID" ]; then
    echo "  Security group already exists: $SG_ID"
else
    echo "  Creating security group..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name $SECURITY_GROUP_NAME \
        --description "Security group for EUR/USD retraining EC2" \
        --vpc-id $VPC_ID \
        --query "GroupId" --output text)
    
    echo "  ✅ Security group created: $SG_ID"
fi

# Note: No inbound rules needed - instance only makes outbound connections

# 3. Launch EC2 Instance
echo -e "\n${GREEN}[3/5] Launching EC2 Instance...${NC}"

# Check if instance already exists
EXISTING_INSTANCE=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=$INSTANCE_NAME" "Name=instance-state-name,Values=running,stopped" \
    --query "Reservations[0].Instances[0].InstanceId" --output text 2>/dev/null)

if [ "$EXISTING_INSTANCE" != "None" ] && [ -n "$EXISTING_INSTANCE" ]; then
    echo -e "${YELLOW}  Instance already exists: $EXISTING_INSTANCE${NC}"
    echo "  The code package has been synchronized to S3."
    echo "  The existing instance will automatically use the new code on its next start."
    INSTANCE_ID=$EXISTING_INSTANCE
else
    echo "  Launching new instance..."
    
    # Encode user data script
    USER_DATA_BASE64=$(base64 -i $USER_DATA_SCRIPT)
    
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id $AMI_ID \
        --instance-type $INSTANCE_TYPE \
        --key-name $KEY_NAME \
        --security-group-ids $SG_ID \
        --iam-instance-profile Name=$IAM_ROLE_NAME \
        --instance-initiated-shutdown-behavior stop \
        --user-data file://$USER_DATA_SCRIPT \
        --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":20,"VolumeType":"gp3"}}]' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME},{Key=Purpose,Value=ModelRetraining}]" \
        --query "Instances[0].InstanceId" --output text)
    
    echo "  ✅ Instance launched: $INSTANCE_ID"
    
    # Wait for instance to be running
    echo "  Waiting for instance to start..."
    aws ec2 wait instance-running --instance-ids $INSTANCE_ID
    
    # Wait for instance to run its course (init script -> train -> poweroff)
    echo "  Instance launched. Waiting for it to run User Data setup, retrain, and shutdown..."
    echo "  This acts as the first verification run."
    
    # Wait for instance to stop (timeout after 20 minutes)
    # Using a loop since 'wait instance-stopped' can hang indefinitely if it doesn't stop
    echo "  Waiting for instance to enter 'stopped' state (Timeout: 20 mins)..."
    
    COUNTER=0
    TIMEOUT=40 # 40 * 30s = 20 mins
    
    while [ $COUNTER -lt $TIMEOUT ]; do
        STATE=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query "Reservations[0].Instances[0].State.Name" --output text)
        
        if [ "$STATE" == "stopped" ]; then
            echo -e "  ✅ Instance has stopped successfully (Task Completed)."
            break
        fi
        
        if [ "$STATE" == "terminated" ] || [ "$STATE" == "shutting-down" ]; then
             echo -e "  ⚠️ Instance terminated unexpectedly."
             break
        fi
        
        echo -n "."
        sleep 30
        ((COUNTER++))
    done
    
    if [ $COUNTER -ge $TIMEOUT ]; then
        echo -e "\n${RED}Timed out waiting for instance to stop.${NC}"
        echo "Check logs: aws s3 ls s3://$S3_BUCKET/logs/"
        exit 1
    fi
fi

# 4. Get instance details
echo -e "\n${GREEN}[4/5] Instance Details:${NC}"
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
    --query "Reservations[0].Instances[0].[InstanceId,InstanceType,State.Name,PrivateIpAddress]" \
    --output table

# 5. Summary
echo -e "\n${GREEN}[5/5] Deployment Summary${NC}"
echo -e "${GREEN}==========================================${NC}"
echo "  ✅ IAM Role: $IAM_ROLE_NAME"
echo "  ✅ Security Group: $SG_ID"
echo "  ✅ EC2 Instance: $INSTANCE_ID"
echo "  ✅ Instance State: Stopped (ready for Lambda trigger)"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Deploy the trigger Lambda:"
echo "     ./scripts/deployment/deploy_lambda_trigger_retrain.sh"
echo ""
echo "  2. Test manual start:"
echo "     aws ec2 start-instances --instance-ids $INSTANCE_ID"
echo ""
echo "  3. Monitor logs (after instance starts):"
echo "     aws logs tail /aws/ec2/retrain --follow"
echo ""
echo "  4. Check S3 for uploaded logs:"
echo "     aws s3 ls s3://$S3_BUCKET/logs/"
echo ""
echo -e "${GREEN}Deployment complete!${NC}"
