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
# NOTE: User Data scripts run as 'root' by default, so 'sudo' is NOT required here.
# If you run these commands manually in the terminal, you MUST use 'sudo'.

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

**Why multiple ports?**
- **Port 80 (HTTP)**: Required so users can access your app without typing a port number (Docker maps this to your app's 8080).
- **Port 443 (HTTPS)**: Required if you later set up SSL/HTTPS.
- **Port 22 (SSH)**: Required for you to manage the server.

```bash
# 1. Get your VPC ID first (Copy the ID, e.g., vpc-12345abcde)
# This command finds the ID of the Default VPC in your current configured region.
VPC_ID=$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)

# Display the result (Copy the ID, e.g., vpc-12345abcde or use the variable $VPC_ID)
echo "Found Default VPC ID: $VPC_ID"

# 2. Create Group (Replace VPC_ID with yours)
aws ec2 create-security-group \
    --group-name eurusd-api-sg \
    --description "Security group for EURUSD ML App Inference API on EC2" \
    --vpc-id $VPC_ID

# 3. Allow SSH
# OPTION A: Rigid Security (Recommended) - Only YOUR current IP can connect.
# if your IP changes (e.g. new wifi), you must update this rule.
MY_IP=$(curl -s checkip.amazonaws.com)
echo "My Public IP is: $MY_IP"

aws ec2 authorize-security-group-ingress \
    --group-name eurusd-api-sg \
    --protocol tcp --port 22 --cidr $MY_IP/32

# OPTION B: Open Access (Easiest) - Allows SSH from ANYWHERE.
# Only safe because we use a Key Pair (pem file). Don't do this with passwords!
aws ec2 authorize-security-group-ingress \
    --group-name eurusd-api-sg \
    --protocol tcp --port 22 --cidr 0.0.0.0/0

# 4. Allow HTTP (Port 80)
aws ec2 authorize-security-group-ingress \
    --group-name eurusd-api-sg \
    --protocol tcp --port 80 --cidr 0.0.0.0/0

# 5. Allow HTTPS (Port 443) - For future SSL support
aws ec2 authorize-security-group-ingress \
    --group-name eurusd-api-sg \
    --protocol tcp --port 443 --cidr 0.0.0.0/0
```

### 3.1 Managing Access Later (Add/Remove IPs)
Security Groups are dynamic. Changes apply immediately.

**To ADD a new IP (e.g. your home IP):**
```bash
aws ec2 authorize-security-group-ingress \
    --group-name eurusd-api-sg \
    --protocol tcp --port 22 --cidr NEW_IP/32
```

**To REMOVE the "Open Access" rule (Lock it down):**
```bash
aws ec2 revoke-security-group-ingress \
    --group-name eurusd-api-sg \
    --protocol tcp --port 22 --cidr 0.0.0.0/0
```

### 3.2 Replacing Security Groups (Cleanup Old One)
If you already launched an instance with a default name like `launch-wizard-2`, follow these steps to switch to the clean `eurusd-api-sg`.

**1. Find your Instance ID:**
```bash
# This gets the ID of your first running instance
INSTANCE_ID=$(aws ec2 describe-instances --filters "Name=instance-state-name,Values=running" --query "Reservations[0].Instances[0].InstanceId" --output text)
echo "Instance ID: $INSTANCE_ID"
```

**2. Find your New Security Group ID:**
```bash
SG_ID=$(aws ec2 describe-security-groups --group-names eurusd-api-sg --query "SecurityGroups[0].GroupId" --output text)
echo "New SG ID: $SG_ID"
```

**3. Attach the New Group (This REPLACES the old one):**
```bash
aws ec2 modify-instance-attribute --instance-id $INSTANCE_ID --groups $SG_ID
```

**4. Delete the Old Group:**
Once detached, you can safely delete the old "launch-wizard" group.
```bash
# Replace 'launch-wizard-2' with the actual name of your old group
aws ec2 delete-security-group --group-name launch-wizard-2
```

### 4. Launch Instance with Bootstrap Script
Save the bootstrap script above as `user_data.sh`.

```bash
aws ec2 run-instances \
    --image-id ami-079db87dc4c10ac91 \
    --count 1 \
    --instance-type t3.medium \
    --key-name your-key-pair-name \
    --security-groups eurusd-api-sg \
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
scp Dockerfile scripts/deploy.sh eurusd:~/
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
