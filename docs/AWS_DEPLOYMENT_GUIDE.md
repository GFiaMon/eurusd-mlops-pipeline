# AWS EC2 Deployment Guide for EUR/USD Flask App

## Overview
This guide covers deploying your Flask ML application to AWS EC2 with two storage options:
1. **Local Storage (EBS)**: Store models and data on EC2's attached EBS volume
2. **S3 Storage**: Store models and data in AWS S3 bucket

## Prerequisites
- AWS Account
- AWS CLI configured locally
- Docker installed locally (for testing)
- EC2 key pair created

---

## Option 1: Local Storage (EBS) - Simpler Approach

### Advantages
- ✅ Simpler setup - no S3 configuration needed
- ✅ Faster access - no network latency
- ✅ Lower costs - no S3 storage/transfer fees
- ✅ Good for development and small-scale deployments

### Disadvantages
- ❌ Models/data tied to specific EC2 instance
- ❌ Harder to share across multiple instances
- ❌ No automatic backups (unless you configure EBS snapshots)
- ❌ Limited by EBS volume size

### Deployment Steps

#### 1. Launch EC2 Instance
```bash
# Choose instance type based on your needs:
# - t3.medium (2 vCPU, 4GB RAM) - minimum for TensorFlow
# - t3.large (2 vCPU, 8GB RAM) - recommended
# - c5.xlarge (4 vCPU, 8GB RAM) - better performance

# AMI: Amazon Linux 2023 or Ubuntu 22.04
# Storage: 20-30 GB EBS volume (gp3 for better performance)
# Security Group: Allow inbound on port 8080 (or 80/443)
```

#### 2. Connect to EC2 Instance
```bash
ssh -i your-key.pem ec2-user@your-ec2-public-ip
```

#### 3. Install Docker on EC2
```bash
# For Amazon Linux 2023
# Install Docker on a EC2 instance
sudo yum update -y
sudo yum install -y docker
# Start the Docker service
sudo systemctl start docker
sudo systemctl enable docker
# Add the 'ec2-user' user to the docker group so you don't need 'sudo'
sudo usermod -a -G docker ec2-user

# For Ubuntu
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ubuntu

# Log out and back in for group changes to take effect
exit
```

#### 4. Transfer Files to EC2
```bash
# From your local machine, create a deployment package
cd /path/to/eurusd-capstone

# Create a tarball with ONLY necessary files
tar -czf eurusd-app.tar.gz \
    api/ \
    models/lstm_trained_model.keras \
    models/lstm_scaler.joblib \
    data/processed/lstm_simple_test_data.csv \
    Dockerfile \
    .dockerignore \
    deploy.sh

# Transfer to EC2
scp -i your-key.pem eurusd-app.tar.gz ec2-user@your-ec2-public-ip:~/
```

#### 5. Build and Run Docker Container on EC2
```bash
# SSH back into EC2
ssh -i your-key.pem ec2-user@your-ec2-public-ip

# Extract files
tar -xzf eurusd-app.tar.gz

# Build Docker image
docker build -t eurusd-predictor .

# Run container
docker run -d \
    --name eurusd-app \
    -p 80:8080 \
    --restart unless-stopped \
    eurusd-predictor

# Check logs
docker logs eurusd-app

# Check if running
docker ps
```

#### 6. Access Your App
```
http://your-ec2-public-ip
```

### Code Changes Required: **NONE**
Your current `api/app.py` works perfectly with local storage!

---

## Option 2: S3 Storage - Production Approach

### Advantages
- ✅ Centralized storage - share across multiple instances
- ✅ Automatic backups and versioning
- ✅ Easy to update models without redeploying
- ✅ Scalable - can handle large files
- ✅ Can use different models for different environments

### Disadvantages
- ❌ More complex setup
- ❌ Additional AWS costs (S3 storage + data transfer)
- ❌ Network latency on first load
- ❌ Requires IAM role configuration

### Deployment Steps

#### 1. Create S3 Bucket
```bash
# Create bucket
aws s3 mb s3://eurusd-ml-models --region us-east-1

# Upload models and data
aws s3 cp models/lstm_trained_model.keras s3://eurusd-ml-models/models/lstm_trained_model.keras
aws s3 cp models/lstm_scaler.joblib s3://eurusd-ml-models/models/lstm_scaler.joblib
aws s3 cp data/processed/lstm_simple_test_data.csv s3://eurusd-ml-models/data/processed/lstm_simple_test_data.csv

# Verify upload
aws s3 ls s3://eurusd-ml-models/ --recursive
```

#### 2. Create IAM Role for EC2
```bash
# Create trust policy file
cat > ec2-trust-policy.json <<EOF
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

# Create IAM role
aws iam create-role \
    --role-name eurusd-ec2-s3-role \
    --assume-role-policy-document file://ec2-trust-policy.json

# Create policy for S3 access
cat > s3-access-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::eurusd-ml-models",
        "arn:aws:s3:::eurusd-ml-models/*"
      ]
    }
  ]
}
EOF

# Attach policy to role
aws iam put-role-policy \
    --role-name eurusd-ec2-s3-role \
    --policy-name s3-read-access \
    --policy-document file://s3-access-policy.json

# Create instance profile
aws iam create-instance-profile \
    --instance-profile-name eurusd-ec2-profile

# Add role to instance profile
aws iam add-role-to-instance-profile \
    --instance-profile-name eurusd-ec2-profile \
    --role-name eurusd-ec2-s3-role
```

#### 3. Launch EC2 with IAM Role
```bash
# When launching EC2, attach the IAM instance profile
# Or attach to existing instance:
aws ec2 associate-iam-instance-profile \
    --instance-id i-your-instance-id \
    --iam-instance-profile Name=eurusd-ec2-profile
```

#### 4. Update Dockerfile for S3
```dockerfile
# Modify the CMD line in Dockerfile to use app_cloud.py
CMD ["python", "api/app_cloud.py"]
```

#### 5. Build and Run with S3 Configuration
```bash
# On EC2, build the image
docker build -t eurusd-predictor .

# Run with S3 environment variables
docker run -d \
    --name eurusd-app \
    -p 80:8080 \
    -e USE_S3=true \
    -e S3_BUCKET=eurusd-ml-models \
    -e S3_MODEL_KEY=models/lstm_trained_model.keras \
    -e S3_SCALER_KEY=models/lstm_scaler.joblib \
    -e S3_DATA_KEY=data/processed/lstm_simple_test_data.csv \
    -e AWS_REGION=us-east-1 \
    --restart unless-stopped \
    eurusd-predictor
```

### Code Changes Required
Use `api/app_cloud.py` instead of `api/app.py` - already created for you!

---

## Comparison: EBS vs S3

| Feature | EBS (Local) | S3 (Cloud) |
|---------|-------------|------------|
| **Setup Complexity** | Simple | Moderate |
| **Cost** | ~$2-5/month | ~$5-10/month |
| **Performance** | Faster (local) | Slower (network) |
| **Scalability** | Limited | Unlimited |
| **Multi-instance** | No | Yes |
| **Model Updates** | Redeploy container | Update S3 file |
| **Backups** | Manual snapshots | Automatic |
| **Best For** | Dev/Testing | Production |

---

## Recommended Approach

### For Development/Testing: Use EBS (Local Storage)
- Use your current `api/app.py`
- No code changes needed
- Simpler and cheaper

### For Production: Use S3
- Use `api/app_cloud.py`
- Better scalability
- Easier model updates

---

## Production Enhancements

### 1. Use Gunicorn Instead of Flask Dev Server
Update Dockerfile CMD:
```dockerfile
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "api.app:app"]
```

### 2. Add NGINX Reverse Proxy
```bash
# Install nginx on EC2
sudo yum install -y nginx

# Configure nginx
sudo tee /etc/nginx/conf.d/eurusd.conf <<EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

sudo systemctl start nginx
sudo systemctl enable nginx
```

### 3. Add SSL Certificate (Let's Encrypt)
```bash
sudo yum install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 4. Set Up CloudWatch Monitoring
```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
sudo rpm -U ./amazon-cloudwatch-agent.rpm
```

### 5. Auto-scaling with Load Balancer
- Create Application Load Balancer
- Create Auto Scaling Group
- Use S3 for shared model storage

---

## Cost Estimates (Monthly)

### EBS Approach
- EC2 t3.medium: ~$30
- EBS 30GB: ~$3
- Data transfer: ~$1
- **Total: ~$34/month**

### S3 Approach
- EC2 t3.medium: ~$30
- EBS 10GB: ~$1
- S3 storage (5GB): ~$0.12
- S3 requests: ~$0.01
- Data transfer: ~$1
- **Total: ~$32/month**

---

## Troubleshooting

### 1. Keras/TensorFlow Error: "Unrecognized keyword arguments: ['batch_shape']"
**Symptom**: Application crashes with `ValueError: Unrecognized keyword arguments: ['batch_shape']`
**Cause**: Version mismatch. You trained the model with TensorFlow 2.16+ (Keras 3) but are deploying with an older version (Keras 2).
**Fix**:
1. Open `api/requirements.txt`
2. Change `tensorflow` version to `2.17.0`
3. Rebuild container:
```bash
docker stop eurusd-app
docker rm eurusd-app
docker build -t eurusd-predictor .
docker run -d --name eurusd-app -p 80:8080 --restart unless-stopped eurusd-predictor
```

### 2. Docker Error: "failed to read dockerfile: no such file"
**Symptom**: `ERROR: failed to solve: failed to read dockerfile: open Dockerfile: no such file or directory`
**Cause**: You are running the command from a subdirectory (like `api/`), but the `Dockerfile` is in the project root.
**Fix**:
```bash
cd ..        # Go up one level to project root
ls -l        # Confirm you see 'Dockerfile'
docker build -t eurusd-predictor .
```

### 3. Container won't start
```bash
docker logs eurusd-app
docker exec -it eurusd-app bash
```

### 4. S3 access denied
```bash
# Check IAM role is attached
aws sts get-caller-identity

# Test S3 access from EC2
aws s3 ls s3://eurusd-ml-models/
```

### 5. Out of memory
```bash
# Check memory usage
docker stats

# Increase EC2 instance size or add swap
sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Next Steps

1. **Choose your storage approach** (EBS or S3)
2. **Test Docker build locally** first
3. **Launch EC2 instance** with appropriate size
4. **Deploy and test**
5. **Add monitoring and alerts**
6. **Set up CI/CD pipeline** (optional)

Need help with any specific step? Let me know!
