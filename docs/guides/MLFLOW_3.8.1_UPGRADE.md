# MLflow 3.8.1 AWS Deployment - Quick Reference

## What Changed

### Script Updates (`scripts/mlops_utils/setup_mlflow_aws.sh`)
- **Python Version**: Upgraded to Python 3.11 (required for MLflow 3.8.1)
- **MLflow Version**: Updated to 3.8.1 (latest)
- **New Dependencies**: Added `prometheus-flask-exporter` for metrics
- **Security Fix**: Added `--allowed-hosts "*"` flag to prevent "Invalid Host header" errors
- **Environment Variable**: Added `prometheus_multiproc_dir=/tmp/mlflow-metrics`

### Key Configuration
```bash
# In systemd service file:
Environment="prometheus_multiproc_dir=/tmp/mlflow-metrics"
ExecStart=/opt/mlflow_venv/bin/mlflow server \
    --backend-store-uri postgresql://... \
    --default-artifact-root s3://eurusd-ml-models/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --serve-artifacts \
    --allowed-hosts "*"
```

## Fresh Deployment

To deploy a new MLflow server with these updates:

```bash
# Run the updated script
./scripts/mlops_utils/setup_mlflow_aws.sh

# The script will output the new IP address
# Update your .env file with the new IP
echo "MLFLOW_TRACKING_URI=http://<NEW-IP>:5000" > .env
```

## Upgrading Existing Instance

If you already have an EC2 instance running and want to upgrade:

### Option 1: SSH and Manual Upgrade
```bash
# SSH into your instance
ssh -i ~/.ssh/aws_keys/eurusd-keypair.pem ec2-user@<YOUR-IP>

# Stop MLflow
sudo systemctl stop mlflow

# Install Python 3.11
sudo dnf install -y python3.11 python3.11-pip python3.11-devel

# Recreate venv with Python 3.11
sudo rm -rf /opt/mlflow_venv
sudo python3.11 -m venv /opt/mlflow_venv
sudo chown -R ec2-user:ec2-user /opt/mlflow_venv
source /opt/mlflow_venv/bin/activate

# Install MLflow 3.8.1
pip install --upgrade pip
pip install mlflow==3.8.1 psycopg2-binary boto3 prometheus-flask-exporter

# Create metrics directory
sudo mkdir -p /tmp/mlflow-metrics
sudo chown root:root /tmp/mlflow-metrics

# Update systemd service
sudo nano /etc/systemd/system/mlflow.service
```

Add these lines to the service file:
```ini
Environment="prometheus_multiproc_dir=/tmp/mlflow-metrics"
```

And update the ExecStart line to include:
```ini
ExecStart=/opt/mlflow_venv/bin/mlflow server \
    --backend-store-uri postgresql://mlflow_user:mlflow_password_123@<DB-ENDPOINT>:5432/mlflow_db \
    --default-artifact-root s3://eurusd-ml-models/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --serve-artifacts \
    --allowed-hosts "*"
```

Then restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart mlflow
sudo systemctl status mlflow
```

### Option 2: Terminate and Redeploy (Recommended)
Your data is safe in RDS and S3, so you can:

```bash
# Terminate the old instance
aws ec2 terminate-instances --instance-ids <OLD-INSTANCE-ID>

# Run the updated setup script
./scripts/mlops_utils/setup_mlflow_aws.sh

# Update .env with new IP
echo "MLFLOW_TRACKING_URI=http://<NEW-IP>:5000" > .env
```

## Local Environment

Update your local environment to match:

```bash
# Install/upgrade MLflow locally
pip install mlflow==3.8.1 boto3 python-dotenv

# Verify version
mlflow --version  # Should show: mlflow, version 3.8.1
```

## Troubleshooting

### "Invalid Host header" Error
- **Cause**: MLflow 3.5.0+ has strict host validation
- **Solution**: The `--allowed-hosts "*"` flag is now included in the script

### Permission Errors During Upgrade
- **Cause**: Virtual environment owned by root
- **Solution**: Use `sudo chown -R ec2-user:ec2-user /opt/mlflow_venv` after creating it

### Prometheus Errors
- **Cause**: Missing `prometheus-flask-exporter` or metrics directory
- **Solution**: Install the package and create `/tmp/mlflow-metrics` directory

## Important Notes

1. **IP Changes**: When you restart EC2, the public IP changes. Consider using an Elastic IP for production.
2. **Data Safety**: All experiments and artifacts are in RDS/S3, so you can safely terminate and recreate the EC2 instance.
3. **Security**: `--allowed-hosts "*"` allows all hosts. For production, restrict this to specific IPs/domains.
4. **Python Version**: MLflow 3.8.1 requires Python 3.10+. We use 3.11 for future compatibility.

## Version Compatibility

| Component | Version | Notes |
|-----------|---------|-------|
| MLflow | 3.8.1 | Latest stable |
| Python | 3.11 | Required for MLflow 3.8.1 |
| PostgreSQL | 15 | RDS backend |
| boto3 | Latest | S3 artifact storage |
| prometheus-flask-exporter | Latest | Metrics support |
