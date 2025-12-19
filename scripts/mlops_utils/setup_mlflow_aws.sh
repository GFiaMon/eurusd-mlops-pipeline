#!/bin/bash
set -e

# --- Configuration ---
PROJECT_TAG="EURUSD_Capstone"
REGION="us-east-1"
S3_BUCKET="eurusd-ml-models"
DB_INSTANCE_IDENTIFIER="eurusd-mlflow-db"
DB_NAME="mlflow_db"
DB_USERNAME="mlflow_user"
DB_PASSWORD="mlflow_password_123" 
EC2_INSTANCE_TYPE="t3.micro" # Verified Free Tier eligible
RDS_INSTANCE_CLASS="db.t3.micro"
# AMI_ID="ami-0c7217cdde317cfec" # OLD (Ubuntu)
# Dynamic lookup for latest Amazon Linux 2023 (AL2023) x86_64 AMI
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-2023.*-x86_64" "Name=state,Values=available" \
    --query "sort_by(Images, &CreationDate)[-1].ImageId" \
    --output text)
KEY_NAME="eurusd-keypair"

echo "🚀 Starting AWS Infrastructure Setup for MLflow (Robust Version)..."

# 0. Check for AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI could not be found. Please install it first."
    exit 1
fi

# 0.1 Robust S3 Check
echo "🪣 Verifying S3 Bucket..."
if ! aws s3api head-bucket --bucket "$S3_BUCKET" 2>/dev/null; then
    echo "❌ Bucket '$S3_BUCKET' not found or no access. Please create it first."
    exit 1
fi
echo "   ✅ Using Bucket: $S3_BUCKET"

# 1. Create Security Groups
echo "🔒 Creating Security Groups..."
VPC_ID=$(aws ec2 describe-vpcs --query "Vpcs[0].VpcId" --output text)
echo "   Using VPC: $VPC_ID"

# MLflow Server SG
SG_NAME="eurusd-mlflow-server-sg"
if ! aws ec2 describe-security-groups --group-names "$SG_NAME" &> /dev/null; then
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "Security group for MLflow Server" \
        --vpc-id "$VPC_ID" \
        --query 'GroupId' \
        --output text)
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 5000 --cidr 0.0.0.0/0
    aws ec2 create-tags --resources "$SG_ID" --tags Key=Project,Value="$PROJECT_TAG"
    echo "   ✅ Created SG: $SG_NAME ($SG_ID)"
else
    SG_ID=$(aws ec2 describe-security-groups --group-names "$SG_NAME" --query "SecurityGroups[0].GroupId" --output text)
    echo "   ℹ️  SG $SG_NAME already exists ($SG_ID)"
fi

# Ensure Ingress Rules (Idempotent)
echo "   🛡️  Ensuring Ingress Rules..."
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0 2>/dev/null || true
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 5000 --cidr 0.0.0.0/0 2>/dev/null || true

# RDS SG
RDS_SG_NAME="eurusd-mlflow-db-sg"
if ! aws ec2 describe-security-groups --group-names "$RDS_SG_NAME" &> /dev/null; then
    RDS_SG_ID=$(aws ec2 create-security-group \
        --group-name "$RDS_SG_NAME" \
        --description "Security group for MLflow RDS" \
        --vpc-id "$VPC_ID" \
        --query 'GroupId' \
        --output text)
    aws ec2 authorize-security-group-ingress --group-id "$RDS_SG_ID" --protocol tcp --port 5432 --source-group "$SG_ID"
    aws ec2 create-tags --resources "$RDS_SG_ID" --tags Key=Project,Value="$PROJECT_TAG"
    echo "   ✅ Created RDS SG: $RDS_SG_NAME ($RDS_SG_ID)"
else
    RDS_SG_ID=$(aws ec2 describe-security-groups --group-names "$RDS_SG_NAME" --query "SecurityGroups[0].GroupId" --output text)
    echo "   ℹ️  RDS SG $RDS_SG_NAME already exists ($RDS_SG_ID)"
fi

# 2. Setup RDS (PostgreSQL)
echo "💾 Setting up RDS (PostgreSQL)..."
if ! aws rds describe-db-instances --db-instance-identifier "$DB_INSTANCE_IDENTIFIER" &> /dev/null; then
    echo "   Creating RDS instance (this may take 5-10 minutes)..."
    aws rds create-db-instance \
        --db-instance-identifier "$DB_INSTANCE_IDENTIFIER" \
        --db-instance-class "$RDS_INSTANCE_CLASS" \
        --engine postgres \
        --master-username "$DB_USERNAME" \
        --master-user-password "$DB_PASSWORD" \
        --db-name "$DB_NAME" \
        --allocated-storage 20 \
        --vpc-security-group-ids "$RDS_SG_ID" \
        --backup-retention-period 0 \
        --no-multi-az \
        --no-publicly-accessible \
        --tags Key=Project,Value="$PROJECT_TAG" \
        --output text > /dev/null
    
    echo "   ⏳ Waiting for RDS to be available (grab a coffee)..."
    aws rds wait db-instance-available --db-instance-identifier "$DB_INSTANCE_IDENTIFIER"
    echo "   ✅ RDS Instance created!"
else
    echo "   ℹ️  RDS Instance $DB_INSTANCE_IDENTIFIER already exists."
    STATUS=$(aws rds describe-db-instances --db-instance-identifier "$DB_INSTANCE_IDENTIFIER" --query "DBInstances[0].DBInstanceStatus" --output text)
    if [ "$STATUS" != "available" ]; then
        echo "   ⏳ Waiting for RDS to be available (current status: $STATUS)..."
        aws rds wait db-instance-available --db-instance-identifier "$DB_INSTANCE_IDENTIFIER"
    fi
fi

# Get RDS Endpoint
DB_ENDPOINT=$(aws rds describe-db-instances --db-instance-identifier "$DB_INSTANCE_IDENTIFIER" --query "DBInstances[0].Endpoint.Address" --output text)
echo "   📌 DB Endpoint: $DB_ENDPOINT"

# 3. Setup IAM Role
echo "🔑 Setting up IAM Role..."
ROLE_NAME="MLflow-EC2-S3-Role"
INSTANCE_PROFILE_NAME="MLflow-EC2-Instance-Profile"

if ! aws iam get-role --role-name "$ROLE_NAME" &> /dev/null; then
    TRUST_POLICY='{
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Principal": { "Service": "ec2.amazonaws.com" },
          "Action": "sts:AssumeRole"
        }
      ]
    }'
    aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document "$TRUST_POLICY" --output text > /dev/null
    aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
    echo "   ✅ Created IAM Role: $ROLE_NAME"
else
    echo "   ℹ️  IAM Role $ROLE_NAME already exists."
fi

if ! aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" &> /dev/null; then
    aws iam create-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" --output text > /dev/null
    aws iam add-role-to-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" --role-name "$ROLE_NAME"
    echo "   ✅ Created Instance Profile: $INSTANCE_PROFILE_NAME"
    sleep 10 # Wait for propagation
else
    echo "   ℹ️  Instance Profile $INSTANCE_PROFILE_NAME already exists."
fi

# 4. Create Key Pair
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" &> /dev/null; then
    echo "   � Creating Key Pair: $KEY_NAME"
    aws ec2 create-key-pair --key-name "$KEY_NAME" --query "KeyMaterial" --output text > "${KEY_NAME}.pem"
    chmod 400 "${KEY_NAME}.pem"
    echo "   ✅ Key Pair saved to ${KEY_NAME}.pem"
else
    echo "   ℹ️  Key Pair $KEY_NAME already exists."
fi

# 5. Launch EC2 (Best Practices: Venv + IMDSv2)
echo "�💻 Launching EC2 Instance (2025 Best Practices)..."

USER_DATA="#!/bin/bash
# Update and install dependencies
dnf update -y
dnf install -y python3-pip python3-devel postgresql15-devel gcc

# Create a virtual environment for MLflow
python3 -m venv /opt/mlflow_venv
source /opt/mlflow_venv/bin/activate
pip install --upgrade pip
pip install mlflow psycopg2-binary boto3

# Create Systemd Service
cat <<EOF > /etc/systemd/system/mlflow.service
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
User=root
WorkingDirectory=/root
ExecStart=/opt/mlflow_venv/bin/mlflow server \\
    --backend-store-uri postgresql://$DB_USERNAME:$DB_PASSWORD@$DB_ENDPOINT:5432/$DB_NAME \\
    --default-artifact-root s3://$S3_BUCKET/mlflow-artifacts \\
    --host 0.0.0.0 \\
    --port 5000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable mlflow
systemctl start mlflow
"

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --count 1 \
    --instance-type "$EC2_INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --iam-instance-profile Name="$INSTANCE_PROFILE_NAME" \
    --user-data "$USER_DATA" \
    --metadata-options "HttpTokens=required" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=MLflow-Server},{Key=Project,Value=$PROJECT_TAG}]" \
    --query "Instances[0].InstanceId" \
    --output text)

echo "   ✅ Instance Launched: $INSTANCE_ID (IMDSv2 Enforced)"
echo "   ⏳ Waiting for instance to be running (to get IP)..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

PUBLIC_DNS=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --query "Reservations[0].Instances[0].PublicDnsName" --output text)

echo ""
echo "🎉 Deployment Complete!"
echo "---------------------------------------------------"
echo "🌍 MLflow Server URL: http://$PUBLIC_DNS:5000"
echo "---------------------------------------------------"
echo "📋 Next Steps:"
echo "1. Create a .env file in your project root:"
echo "   echo \"MLFLOW_TRACKING_URI=http://$PUBLIC_DNS:5000\" > .env"
echo "2. Install python-dotenv:"
echo "   pip install python-dotenv"
echo "3. Run your training script:"
echo "   python src/03_train_models.py"