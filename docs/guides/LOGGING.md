# 📋 System Logging Guide

This guide explains how logging is handled in the EUR/USD MLOps pipeline and how to access logs via the AWS CLI.

## 1. Overview

The system centralizes all logs in S3. This ensures that even ephemeral compute resources (Lambda, EC2) leave a permanent record of their execution.

| Component | Log Source | Log Destination | Filename Pattern |
| :--- | :--- | :--- | :--- |
| **Data Ingestion** | AWS Lambda | `s3://<bucket>/logs/` | `ingest_YYYYMMDD_HHMMSS.log` |
| **Model Retraining** | EC2 Worker | `s3://<bucket>/logs/` | `retrain_YYYYMMDD_HHMMSS.log` |

---

## 2. Log Structure

### Ingestion Logs (Lambda)
The ingestion logs capture the output of the data download and S3 upload process.
*   **Trigger**: Daily at 22:00 UTC (EventBridge).
*   **Contents**: `yfinance` download status, DataManager operations, and save confirmation.

### Retraining Logs (EC2)
The retraining logs capture the entire Docker execution on the training worker.
*   **Trigger**: Daily at 22:10 UTC (EventBridge -> Lambda -> EC2 Start).
*   **Contents**: Docker start/stop events, MLflow training metrics, and error traces.

---

## 3. How to Check Logs (CLI)

You can manage logs entirely from your terminal using `aws s3` commands.

### List All Logs
To see all available log files:
```bash
aws s3 ls s3://eurusd-ml-models/logs/
```

### View the Latest Log (Quickest Method)
To view the content of a specific log file without downloading it to a file on disk (streams directly to `less`):
```bash
# Syntax: aws s3 cp s3://BUFFER/KEY - | less

# Example:
aws s3 cp s3://eurusd-ml-models/logs/ingest_20260106_163653.log - | less
```
*   Press `q` to exit `less`.

### Download a Log
If you want to keep the file or analyze it locally:
```bash
# Download to current directory
aws s3 cp s3://eurusd-ml-models/logs/ingest_20260106_163653.log .

# Open with your preferred editor
code ingest_20260106_163653.log
```

## 4. Troubleshooting

**"I don't see a log file for today"**
1.  **Ingestion**: Check AWS CloudWatch Logs for the Lambda function `eurusd-ingest`. If the Lambda crashed *before* the upload step (e.g., timeout or out of memory), the S3 log upload might have failed.
2.  **Retraining**: Check the EC2 Instance Console. If the instance didn't start (e.g., "Insufficient Instance Capacity"), no log will be generated.

**"The log is incomplete"**
*   **Retraining**: The log is uploaded at the very end of the `user_data` script. If the EC2 instance crashes or force-terminates mid-script, the log upload might be skipped.
