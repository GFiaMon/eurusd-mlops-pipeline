# MLflow AWS Deployment Guide

This guide outlines how to deploy a production-ready MLflow Tracking Server on AWS and how to configure your local environment to use it.

## Architecture: Remote Tracking Server

We will use the **Remote Tracking Server** scenario with **S3 for Artifact Storage** and **RDS (PostgreSQL) for the Backend Store**.

-   **Tracking Server (EC2)**: Hosts the MLflow UI/API.
-   **Backend Store (RDS)**: Stores experiment metadata (runs, parameters, metrics).
-   **Artifact Store (S3)**: Stores model artifacts (files, plots, models).
-   **Local Client**: Your laptop running the training scripts.

```mermaid
graph LR
    LocalMachine[Local Machine] -->|Logs Metrics/Params| EC2[EC2 - MLflow Server]
    LocalMachine -->|Uploads Artifacts| S3[S3 - Artifact Store]
    EC2 -->|Reads/Writes Metadata| RDS[RDS - PostgreSQL]
    EC2 -->|Proxies Artifacts (Optional)| S3
```

> [!NOTE]
> In this setup, your local machine needs direct access to S3 to upload artifacts, or you can configure the Tracking Server to proxy artifact requests. For simplicity, we assume direct access (standard MLflow behavior).

---

## Step 1: Create AWS Resources

### 1.1 S3 Bucket (Artifact Store)
1.  Go to **S3** Console.
2.  Create a bucket (e.g., `eurusd-ml-models`).
3.  Block all public access.

### 1.2 RDS Instance (Backend Store)
1.  Go to **RDS** Console.
2.  Create a **PostgreSQL** database.
    -   **Template**: Free Tier (if eligible) or Dev/Test.
    -   **Master Username**: `mlflow_user`
    -   **Password**: [SECURE_PASSWORD]
3.  **Public Access**: Yes (if you want to assist debugging easily, but ideally No + VPN/Bastion). For this guide/MVP, ensure your EC2 instance can talk to it (same VPC/Security Group).
4.  Create a database name: `mlflow_db` (in "Additional Configuration").

### 1.3 EC2 Instance (Tracking Server)
1.  Launch an **Ubuntu** `t2.micro` instance.
2.  **Security Group**:
    -   Allow **SSH (22)** from your IP.
    -   Allow **Custom TCP (5000)** from Anywhere (0.0.0.0/0) or your IP.

---

## Step 2: Configure the Tracking Server (EC2)

SSH into your EC2 instance:
```bash
ssh -i "your-key.pem" ubuntu@<ec2-public-ip>
```

### 2.1 Install Dependencies
```bash
sudo apt update
sudo apt install python3-pip python3-venv libpq-dev -y
```

### 2.2 Setup MLflow Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install mlflow boto3 psycopg2-binary
```

### 2.3 Configure AWS Credentials
The EC2 instance needs permission to access S3.
**Best Practice**: Attach an **IAM Role** to the EC2 instance with `AmazonS3FullAccess` (or scoped policy).

If using keys (not recommended for production):
```bash
# pip install awscli
# aws configure
```

### 2.4 Start MLflow Server
Run the server pointing to RDS and S3.
```bash
# Database URL format: postgresql://<user>:<password>@<host>:<port>/<dbname>
export DB_URI="postgresql://mlflow_user:password@rds-endpoint:5432/mlflow_db"

mlflow server \
    --backend-store-uri $DB_URI \
    --default-artifact-root s3://mlflow-artifacts-eurusd-capstone/ \
    --host 0.0.0.0 \
    --port 5000
```
*Tip: Use `nohup` or `systemd` to keep this running in the background.*

---

## Step 3: Configure Local Client (Your Laptop)

To send runs to this remote server, you simply need to point MLflow to it.

### 3.1 Environment Variables
Set the tracking URI in your terminal (or `.env` file):
```bash
export MLFLOW_TRACKING_URI="http://<ec2-public-ip>:5000"
```

### 3.2 AWS Access
Your local machine needs permission to write to the S3 bucket. Ensure you have `~/.aws/credentials` configured or `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` set.

### 3.3 Verify Connection
Run your code:
```bash
python src/03_train_models.py
```
Check the UI at `http://<ec2-public-ip>:5000`. You should see your experiments appearing there.

---

## Adapting `03_train_models.py`
The code is already compatible!
```python
# src/03_train_models.py
# ...
# MLflow automatically picks up the MLFLOW_TRACKING_URI env var.
# If not set, it defaults to ./mlruns (local).
mlflow.set_experiment(EXPERIMENT_NAME)
# ...
```

> [!IMPORTANT]
> **PMDARIMA & Logging**:
> We log the `arima_model` object using `mlflow.sklearn.log_model`.
> While `auto_arima` determines parameters dynamically, logging the fitted model object is crucial for deployment pipelines (e.g., AWS SageMaker or Lambda) where you want to `load_model` and `predict` without re-fitting.
