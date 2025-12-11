# 🛠️ EC2 Detailed Setup & SSH Configuration Guide

This guide complements the Deployment Guide by providing step-by-step instructions for infrastructure setup.

## 📚 Table of Contents
1. [Method 1: AWS Console Setup](#method-1-aws-console-setup)
2. [Method 2: AWS CLI Setup (Command Line)](#method-2-aws-cli-setup)
3. [Bootstrapping (Auto-Install Docker)](#bootstrapping-auto-install-docker)
4. [SSH Configuration (The Easy Way)](#ssh-configuration-simpler-access)

---

## 🚀 Bootstrapping (Auto-Install Docker)

**What is Bootstrapping?**
When an EC2 instance launches, it can look for a "User Data" script to run immediately. We can use this to install Docker, Git, and setting permissions *automatically*, so you don't have to SSH in and run manual commands.

### The User Data Script
Copy this script. You will use it in both Method 1 (Console) and Method 2 (CLI).

```bash
#!/bin/bash
# Install updates
yum update -y

# Install Docker
yum install -y docker

# Start Docker service
systemctl start docker
systemctl enable docker

# Add ec2-user to docker group (allows running docker without sudo)
usermod -a -G docker ec2-user

# Install Git (optional, useful for debugging)
yum install -y git

# Create app directory
mkdir -p /home/ec2-user/app
chown ec2-user:ec2-user /home/ec2-user/app

echo "✅ Bootstrapping complete! Docker is ready."
```

---

## Method 1: AWS Console Setup

### 1. Launch Instance
1. Log in to [AWS Console](https://console.aws.amazon.com/ec2).
2. Click **"Launch Instance"** (Orange button).
3. **Name**: `eurusd-predictor`
4. **OS (AMI)**: Select **Amazon Linux 2023** (Free tier eligible).
5. **Instance Type**: Select **t3.medium** (Recommended for ML) or `t2.micro` (Free tier, but might crash with memory errors).
6. **Key Pair**: Select your existing key (e.g., `my-mac-key`) or create new.

### 2. Network Settings
1. Click **"Edit"** near Network Settings.
2. **Security Group**: "Create security group".
3. **Inbound Rules**:
   - Port 22 (SSH) - Source: `My IP` (For security).
   - Port 80 (HTTP) - Source: `Anywhere` (0.0.0.0/0).
   - Port 8080 (Custom TCP) - Source: `Anywhere` (Optional, if testing directly).

### 3. Advanced Details (Bootstrapping)
1. Scroll down to **"Advanced details"**.
2. Scroll to the very bottom to **"User data"**.
3. Paste the **User Data Script** from the section above.

### 4. Launch
1. Click **"Launch instance"**.
2. Wait ~2 minutes for "2/2 checks passed" and for the script to finish installing Docker.

---

## Method 2: AWS CLI Setup

If you prefer the terminal, you can launch everything with one command.

### 1. Prerequisites
- AWS CLI installed (`brew install awscli`)
- Configured (`aws configure`)
- Key pair file (`ironhack-key.pem`) locally available.

### 2. Get your VPC and Subnet IDs (Optional but recommended)
```bash
aws ec2 describe-vpcs
aws ec2 describe-subnets
```

### 3. Create Security Group
```bash
# Create Group
aws ec2 create-security-group \
    --group-name eurusd-sg \
    --description "Security group for EURUSD ML App"

# Allow SSH (Replace YOUR_IP with your actual IP, e.g., 123.45.67.89/32)
aws ec2 authorize-security-group-ingress \
    --group-name eurusd-sg \
    --protocol tcp --port 22 --cidr YOUR_IP/32

# Allow HTTP
aws ec2 authorize-security-group-ingress \
    --group-name eurusd-sg \
    --protocol tcp --port 80 --cidr 0.0.0.0/0
```

### 4. Launch Instance with Bootstrap Script
Save the bootstrap script above as `user_data.sh`.

```bash
aws ec2 run-instances \
    --image-id ami-079db87dc4c10ac91 \
    --count 1 \
    --instance-type t3.medium \
    --key-name your-key-pair-name \
    --security-groups eurusd-sg \
    --user-data file://user_data.sh \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=eurusd-predictor}]'
```
*(Note: `ami-079db87dc4c10ac91` is Amazon Linux 2023 in us-east-1. Change ID if using a different region).*

---

## 🔐 SSH Configuration (Simpler Access)

Instead of typing this every time:
`ssh -i /path/to/key.pem ec2-user@54.123.45.67`

You can type just:
`ssh eurusd`

### 1. Edit Config File
Open (or create) your SSH config file:
```bash
nano ~/.ssh/config
```

### 2. Add Host Entry
Add this block at the bottom:

```ssh
Host eurusd
    HostName 54.123.45.67      # Replace with your EC2 Public IP
    User ec2-user
    IdentityFile ~/.ssh/your-key.pem
    ServerAliveInterval 60
```

### 3. Permissions
Ensure correct permissions (SSH is strict about this):
```bash
chmod 600 ~/.ssh/config
chmod 400 ~/.ssh/your-key.pem
```

### 4. Connect
Now you can simply run:
```bash
ssh eurusd
```

### 5. Deploy with Alias
You can even use this alias with the deploy script if you modify the SCP command slightly, or just for copying files:
```bash
scp Dockerfile deploy.sh eurusd:~/
```

---

## 🔍 Verification

After launching your instance with the bootstrap script, verify Docker is installed:

```bash
# SSH into instance
ssh eurusd

# Check Docker version
docker --version

# Check if you can run docker without sudo
docker ps
```

If `docker ps` works without `sudo`, your bootstrapping was successful!
