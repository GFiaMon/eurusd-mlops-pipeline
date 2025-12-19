#!/bin/bash
set -e

# provision_api_ec2.sh - Provision EC2 instance for Flask API serving ML models from MLflow
# Based on successful setup_mlflow_aws.sh pattern

# --- Configuration ---
PROJECT_TAG="EURUSD_Capstone"
REGION="us-east-1"
S3_BUCKET="eurusd-ml-models"
EC2_INSTANCE_TYPE="t3.micro"  # Free Tier eligible
KEY_NAME="eurusd-inference-app"  # Existing key pair
IAM_ROLE_NAME="eurusd-ec2-s3-role"  # Will reuse if exists
INSTANCE_PROFILE_NAME="eurusd-api-instance-profile"
SG_NAME="eurusd-flask-api-sg"
API_PORT=8080

echo "🚀 Starting EC2 Flask API Provisioning..."
echo "========================================"

# 0. Pre-flight Checks
echo ""
echo "✈️  Pre-flight Checks..."

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install it first."
    exit 1
fi
echo "   ✅ AWS CLI installed"

# Check S3 bucket
echo "   🪣 Verifying S3 Bucket..."
if ! aws s3api head-bucket --bucket "$S3_BUCKET" 2>/dev/null; then
    echo "   ❌ Bucket '$S3_BUCKET' not found or no access."
    exit 1
fi
echo "   ✅ S3 Bucket verified: $S3_BUCKET"

# Check key pair and local file with comprehensive management
echo "   🔑 Managing Key Pair..."

# Default local key path
LOCAL_KEY_PATH="$HOME/.ssh/aws_keys/${KEY_NAME}.pem"
KEY_DIR="$(dirname "$LOCAL_KEY_PATH")"

# Ensure key directory exists
mkdir -p "$KEY_DIR"

# Check if key pair exists in AWS
KEY_EXISTS_IN_AWS=false
if aws ec2 describe-key-pairs --key-names "$KEY_NAME" &> /dev/null; then
    KEY_EXISTS_IN_AWS=true
    echo "   ✅ Key pair exists in AWS: $KEY_NAME"
else
    echo "   ⚠️  Key pair '$KEY_NAME' not found in AWS"
fi

# Check if local key file exists
if [ -f "$LOCAL_KEY_PATH" ]; then
    echo "   ✅ Local key file found: $LOCAL_KEY_PATH"
    
    # Check and fix permissions
    CURRENT_PERMS=$(stat -f "%OLp" "$LOCAL_KEY_PATH" 2>/dev/null || stat -c "%a" "$LOCAL_KEY_PATH" 2>/dev/null)
    if [ "$CURRENT_PERMS" != "400" ] && [ "$CURRENT_PERMS" != "600" ]; then
        echo "   ⚠️  Fixing key file permissions (was $CURRENT_PERMS, setting to 400)..."
        chmod 400 "$LOCAL_KEY_PATH"
        echo "   ✅ Permissions fixed"
    else
        echo "   ✅ Key file permissions are correct ($CURRENT_PERMS)"
    fi
    
    # Verify key pair exists in AWS
    if [ "$KEY_EXISTS_IN_AWS" = false ]; then
        echo "   ❌ Local key file exists but AWS key pair is missing!"
        echo "   This means the key pair was deleted from AWS."
        read -p "   Delete local file and create new key pair? (y/n): " RECREATE_KEY
        if [ "$RECREATE_KEY" != "y" ]; then
            echo "   ❌ Cannot proceed without valid key pair"
            exit 1
        fi
        rm -f "$LOCAL_KEY_PATH"
        echo "   🗑️  Deleted local key file"
    fi
else
    echo "   ⚠️  Local key file not found at $LOCAL_KEY_PATH"
    
    # Prompt for alternate location
    read -p "   Do you have the .pem file in another location? (y/n): " HAS_KEY_ELSEWHERE
    
    if [ "$HAS_KEY_ELSEWHERE" = "y" ]; then
        read -p "   Enter full path to .pem file: " ALT_KEY_PATH
        if [ -f "$ALT_KEY_PATH" ]; then
            # Copy to standard location
            cp "$ALT_KEY_PATH" "$LOCAL_KEY_PATH"
            chmod 400 "$LOCAL_KEY_PATH"
            echo "   ✅ Copied key file to $LOCAL_KEY_PATH"
        else
            echo "   ❌ File not found at $ALT_KEY_PATH"
            exit 1
        fi
    else
        # Need to create new key pair
        if [ "$KEY_EXISTS_IN_AWS" = true ]; then
            echo "   ⚠️  Key pair exists in AWS but local file is missing"
            echo "   To create a new .pem file, we need to delete and recreate the key pair in AWS"
            read -p "   Delete AWS key pair '$KEY_NAME' and create new one? (y/n): " DELETE_AND_RECREATE
            
            if [ "$DELETE_AND_RECREATE" = "y" ]; then
                echo "   🗑️  Deleting existing key pair from AWS..."
                aws ec2 delete-key-pair --key-name "$KEY_NAME"
                KEY_EXISTS_IN_AWS=false
            else
                echo "   ❌ Cannot proceed without local key file"
                exit 1
            fi
        fi
        
        # Create new key pair
        if [ "$KEY_EXISTS_IN_AWS" = false ]; then
            echo "   🔑 Creating new key pair: $KEY_NAME"
            aws ec2 create-key-pair \
                --key-name "$KEY_NAME" \
                --query "KeyMaterial" \
                --output text > "$LOCAL_KEY_PATH"
            
            chmod 400 "$LOCAL_KEY_PATH"
            echo "   ✅ New key pair created and saved to $LOCAL_KEY_PATH"
            echo "   📝 Permissions set to 400"
        fi
    fi
fi

# Final verification
if [ ! -f "$LOCAL_KEY_PATH" ]; then
    echo "   ❌ Key file still not found after all attempts"
    exit 1
fi

echo "   ✅ Key pair ready: $KEY_NAME"
echo "   ✅ Local key file: $LOCAL_KEY_PATH"

# Discover MLflow Server IP
echo ""
echo "🔍 Discovering MLflow Server..."
MLFLOW_INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=MLflow-Server" "Name=instance-state-name,Values=running,stopped" \
    --query "Reservations[0].Instances[0].InstanceId" \
    --output text 2>/dev/null)

if [ "$MLFLOW_INSTANCE_ID" == "None" ] || [ -z "$MLFLOW_INSTANCE_ID" ]; then
    echo "   ❌ No EC2 instance found with tag Name=MLflow-Server"
    read -p "   Enter MLflow server URI manually (e.g., http://IP:5000): " MLFLOW_TRACKING_URI
else
    echo "   ✅ Found MLflow instance: $MLFLOW_INSTANCE_ID"
    
    # Check state
    MLFLOW_STATE=$(aws ec2 describe-instances \
        --instance-ids "$MLFLOW_INSTANCE_ID" \
        --query "Reservations[0].Instances[0].State.Name" \
        --output text)
    
    echo "   📊 Instance state: $MLFLOW_STATE"
    
    if [ "$MLFLOW_STATE" != "running" ]; then
        echo "   ⚠️  MLflow server is not running (state: $MLFLOW_STATE)"
        read -p "   Start the instance now? (y/n): " START_MLFLOW
        
        if [ "$START_MLFLOW" == "y" ]; then
            echo "   🔄 Starting MLflow instance..."
            aws ec2 start-instances --instance-ids "$MLFLOW_INSTANCE_ID" > /dev/null
            echo "   ⏳ Waiting for instance to be running..."
            aws ec2 wait instance-running --instance-ids "$MLFLOW_INSTANCE_ID"
            echo "   ✅ MLflow instance started"
        else
            read -p "   Enter MLflow server URI manually: " MLFLOW_TRACKING_URI
        fi
    fi
    
    # Get public IP if running
    if [ -z "$MLFLOW_TRACKING_URI" ]; then
        MLFLOW_IP=$(aws ec2 describe-instances \
            --instance-ids "$MLFLOW_INSTANCE_ID" \
            --query "Reservations[0].Instances[0].PublicIpAddress" \
            --output text)
        
        if [ "$MLFLOW_IP" == "None" ] || [ -z "$MLFLOW_IP" ]; then
            echo "   ❌ Could not get public IP for MLflow server"
            read -p "   Enter MLflow server URI manually: " MLFLOW_TRACKING_URI
        else
            MLFLOW_TRACKING_URI="http://${MLFLOW_IP}:5000"
            echo "   ✅ MLflow URI: $MLFLOW_TRACKING_URI"
        fi
    fi
fi

# Verify MLflow is accessible (optional, requires curl)
echo "   🌐 Testing MLflow connectivity..."
if curl -sf --max-time 5 "$MLFLOW_TRACKING_URI/health" > /dev/null 2>&1 || \
   curl -sf --max-time 5 "$MLFLOW_TRACKING_URI" > /dev/null 2>&1; then
    echo "   ✅ MLflow server is accessible"
else
    echo "   ⚠️  Could not verify MLflow connectivity (might be OK if just started)"
fi

# Check local files
echo ""
echo "📦 Verifying local deployment files..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

FILES_TO_CHECK=("Dockerfile.cloud" "scripts/deploy.sh" "api/app_cloud.py")
for file in "${FILES_TO_CHECK[@]}"; do
    if [ ! -f "$PROJECT_ROOT/$file" ]; then
        echo "   ❌ Required file not found: $file"
        exit 1
    fi
done
echo "   ✅ All required files present"

# 1. IAM Role Configuration
echo ""
echo "🔑 Setting up IAM Role..."

# Check if role exists
if aws iam get-role --role-name "$IAM_ROLE_NAME" &> /dev/null; then
    echo "   ✅ IAM role '$IAM_ROLE_NAME' already exists - reusing"
else
    echo "   Creating new IAM role: $IAM_ROLE_NAME"
    
    # Trust policy
    TRUST_POLICY='{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": { "Service": "ec2.amazonaws.com" },
        "Action": "sts:AssumeRole"
      }]
    }'
    
    aws iam create-role \
        --role-name "$IAM_ROLE_NAME" \
        --assume-role-policy-document "$TRUST_POLICY" \
        --output text > /dev/null
    
    # Minimal S3 policy (read-only)
    S3_POLICY='{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Action": [
          "s3:GetObject",
          "s3:ListBucket"
        ],
        "Resource": [
          "arn:aws:s3:::'"$S3_BUCKET"'",
          "arn:aws:s3:::'"$S3_BUCKET"'/*"
        ]
      }]
    }'
    
    aws iam put-role-policy \
        --role-name "$IAM_ROLE_NAME" \
        --policy-name "S3ReadOnly" \
        --policy-document "$S3_POLICY"
    
    echo "   ✅ Created IAM role with minimal S3 permissions"
fi

# Instance profile
if ! aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" &> /dev/null; then
    echo "   Creating instance profile..."
    aws iam create-instance-profile \
        --instance-profile-name "$INSTANCE_PROFILE_NAME" \
        --output text > /dev/null
    
    aws iam add-role-to-instance-profile \
        --instance-profile-name "$INSTANCE_PROFILE_NAME" \
        --role-name "$IAM_ROLE_NAME"
    
    echo "   ✅ Created instance profile"
    echo "   ⏳ Waiting for IAM propagation (10s)..."
    sleep 10
else
    echo "   ✅ Instance profile already exists"
fi

# 2. Security Group Setup
echo ""
echo "🔒 Setting up Security Group..."

VPC_ID=$(aws ec2 describe-vpcs --query "Vpcs[0].VpcId" --output text)
echo "   Using VPC: $VPC_ID"

if ! aws ec2 describe-security-groups --group-names "$SG_NAME" &> /dev/null; then
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "Security group for Flask API server" \
        --vpc-id "$VPC_ID" \
        --query 'GroupId' \
        --output text)
    
    # Ingress rules
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 80 --cidr 0.0.0.0/0
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 443 --cidr 0.0.0.0/0
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port "$API_PORT" --cidr 0.0.0.0/0
    
    aws ec2 create-tags --resources "$SG_ID" --tags Key=Project,Value="$PROJECT_TAG"
    echo "   ✅ Created SG: $SG_NAME ($SG_ID)"
    echo "   📝 Opened ports: 22, 80, 443, $API_PORT"
else
    SG_ID=$(aws ec2 describe-security-groups --group-names "$SG_NAME" --query "SecurityGroups[0].GroupId" --output text)
    echo "   ✅ Security group already exists: $SG_ID"
fi

# Ensure ingress rules (idempotent)
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0 2>/dev/null || true
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 80 --cidr 0.0.0.0/0 2>/dev/null || true
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 443 --cidr 0.0.0.0/0 2>/dev/null || true
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port "$API_PORT" --cidr 0.0.0.0/0 2>/dev/null || true

# 3. Get AMI ID
echo ""
echo "🖥️  Getting Amazon Linux 2023 AMI..."
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-2023.*-x86_64" "Name=state,Values=available" \
    --query "sort_by(Images, &CreationDate)[-1].ImageId" \
    --output text)

if [ -z "$AMI_ID" ] || [ "$AMI_ID" == "None" ]; then
    echo "   ❌ Could not find Amazon Linux 2023 AMI"
    exit 1
fi
echo "   ✅ Using AMI: $AMI_ID"

# 4. Create user data script
echo ""
echo "📝 Preparing user data script..."

USER_DATA="#!/bin/bash
# Bootstrap script for Flask API EC2 instance
dnf update -y
dnf install -y docker git

# Start and enable Docker
systemctl enable docker
systemctl start docker

# Add ec2-user to docker group
usermod -a -G docker ec2-user

# Create deployment directory
mkdir -p /home/ec2-user/eurusd-app/{api,data,scripts}
chown -R ec2-user:ec2-user /home/ec2-user/eurusd-app

# Log completion
echo 'Bootstrap complete' > /home/ec2-user/bootstrap.log
"

# 5. Launch EC2 Instance
echo ""
echo "🚀 Launching EC2 instance..."

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --count 1 \
    --instance-type "$EC2_INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --iam-instance-profile Name="$INSTANCE_PROFILE_NAME" \
    --user-data "$USER_DATA" \
    --metadata-options "HttpTokens=required" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=eurusd-api-server},{Key=Project,Value=$PROJECT_TAG}]" \
    --query "Instances[0].InstanceId" \
    --output text)

echo "   ✅ Instance launched: $INSTANCE_ID"
echo "   ⏳ Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text)

PUBLIC_DNS=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query "Reservations[0].Instances[0].PublicDnsName" \
    --output text)

echo "   ✅ Instance running!"
echo "   📍 Public IP: $PUBLIC_IP"
echo "   📍 Public DNS: $PUBLIC_DNS"

# Wait for status checks
echo "   ⏳ Waiting for status checks to pass (may take 2-3 minutes)..."
aws ec2 wait instance-status-ok --instance-ids "$INSTANCE_ID"
echo "   ✅ Status checks passed"

# Wait for user data script to complete
echo "   ⏳ Waiting for bootstrap to complete (30s)..."
sleep 30

# Verify Docker installation (user data can fail silently)
echo ""
echo "🐳 Verifying Docker installation..."
DOCKER_CHECK=$(ssh -i "$LOCAL_KEY_PATH" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    -o ConnectTimeout=10 \
    ec2-user@${PUBLIC_IP} \
    "command -v docker && sudo systemctl is-active docker" 2>/dev/null || echo "FAILED")

if echo "$DOCKER_CHECK" | grep -q "active"; then
    echo "   ✅ Docker is installed and running"
else
    echo "   ⚠️  Docker not found or not running - installing manually..."
    echo "   (User data bootstrap may have failed)"
    
    # Install Docker manually via SSH
    ssh -i "$LOCAL_KEY_PATH" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR \
        ec2-user@${PUBLIC_IP} \
        "sudo dnf update -y && \
         sudo dnf install -y docker git && \
         sudo systemctl enable docker && \
         sudo systemctl start docker && \
         sudo usermod -a -G docker ec2-user && \
         mkdir -p /home/ec2-user/eurusd-app/{api,data,scripts}"
    
    # Verify again
    sleep 3
    DOCKER_RETRY=$(ssh -i "$LOCAL_KEY_PATH" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR \
        ec2-user@${PUBLIC_IP} \
        "sudo systemctl is-active docker" 2>/dev/null || echo "FAILED")
    
    if echo "$DOCKER_RETRY" | grep -q "active"; then
        echo "   ✅ Docker installed successfully via fallback method"
    else
        echo "   ❌ Failed to install Docker. Please check the instance manually."
        echo "   SSH: ssh -i $LOCAL_KEY_PATH ec2-user@${PUBLIC_IP}"
        exit 1
    fi
fi

# 6. Deploy Application
echo ""
echo "📦 Deploying application..."

# Create temporary deployment package
TEMP_DIR=$(mktemp -d)
DEPLOY_PACKAGE="$TEMP_DIR/deploy-package.tar.gz"

cd "$PROJECT_ROOT"
tar -czf "$DEPLOY_PACKAGE" \
    Dockerfile.cloud \
    api/ \
    data/ \
    scripts/deploy.sh

echo "   ✅ Created deployment package"

# SCP to EC2
echo "   📤 Uploading to EC2..."
scp -i "$LOCAL_KEY_PATH" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    "$DEPLOY_PACKAGE" \
    ec2-user@${PUBLIC_IP}:/home/ec2-user/eurusd-app/

echo "   ✅ Upload complete"

# Extract and deploy
echo "   🚀 Running deployment script on EC2..."
ssh -i "$LOCAL_KEY_PATH" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    ec2-user@${PUBLIC_IP} \
    "cd /home/ec2-user/eurusd-app && \
     tar -xzf deploy-package.tar.gz && \
     bash scripts/deploy.sh s3 $S3_BUCKET $MLFLOW_TRACKING_URI"

# Cleanup
rm -rf "$TEMP_DIR"

# 7. Validation
echo ""
echo "🧪 Testing deployment..."
sleep 5

HEALTH_RESPONSE=$(curl -sf http://${PUBLIC_IP}:${API_PORT}/health || echo "FAILED")

if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo "   ✅ Health check passed!"
else
    echo "   ⚠️  Health check returned: $HEALTH_RESPONSE"
    echo "   (Container may still be starting)"
fi

# 8. Summary
echo ""
echo "🎉 Deployment Complete!"
echo "============================================"
echo "📋 Instance Details:"
echo "   Instance ID: $INSTANCE_ID"
echo "   Public IP: $PUBLIC_IP"
echo "   Public DNS: $PUBLIC_DNS"
echo ""
echo "🌐 Application URLs:"
echo "   Flask API: http://${PUBLIC_IP}:${API_PORT}"
echo "   Health: http://${PUBLIC_IP}:${API_PORT}/health"
echo "   Predict: http://${PUBLIC_IP}:${API_PORT}/api/predict"
echo ""
echo "🔗 MLflow Server:"
echo "   $MLFLOW_TRACKING_URI"
echo ""
echo "🔑 SSH Access:"
echo "   ssh -i $LOCAL_KEY_PATH ec2-user@${PUBLIC_IP}"
echo ""
echo "💡 Useful Commands:"
echo "   View logs: ssh ... 'docker logs -f eurusd-app'"
echo "   Restart: ssh ... 'docker restart eurusd-app'"
echo "   Stop: ssh ... 'docker stop eurusd-app'"
echo "   Check status: curl http://${PUBLIC_IP}:${API_PORT}/health"
echo ""
echo "📝 Next Steps:"
echo "1. Test the API:"
echo "   curl http://${PUBLIC_IP}:${API_PORT}/api/predict"
echo "2. Monitor logs for any issues"
echo "3. Update DNS/Load Balancer if needed"
echo "============================================"
