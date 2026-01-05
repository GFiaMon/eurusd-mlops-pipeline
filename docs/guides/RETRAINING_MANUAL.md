# EUR/USD Daily Model Retraining Manual

This manual provides a complete guide to the automated retraining system, including architecture decisions, deployment steps, and daily monitoring.

## 1. Architecture Overview

- **Strategy**: **Start/Stop** (Persistent Worker)
- **Why**: 
  - **Stability**: The instance maintains its configuration. We don't risk "insufficient capacity" errors that can happen when requesting new spot instances daily.
  - **Simplicity**: No complex Launch Templates or Lambda auto-scaling logic. The triggering Lambda is extremely simple.
  - **Cost**: The only "extra" cost vs. terminating is the EBS volume storage (~$1.60/month). The compute cost is identical ($0/hr when stopped).

### Workflow
1. **22:10 UTC** (Mon-Fri): EventBridge triggers Lambda.
2. **Lambda**: Finds instance with tag `Name=eurusd-retrain-worker` and starts it.
3. **EC2 Boots**:
   - `cloud-init` runs `/usr/local/bin/eurusd-retrain.sh` (Persistent Per-Boot Script).
   - Updates Docker & pulls latest image (if changed).
   - Runs container: Retrains 3 models (Champion/Challenger/Candidate).
   - Uploads logs to S3.
   - **Self-Terminates**: Runs `sudo poweroff` to stop itself.

## 2. Daily Monitoring (How do I know it worked?)

You don't need to check every day, but if you want to verify:

### Option A: Check the Logs (Recommended)
The system uploads a log file to S3 after every run.

```bash
# List recent logs
aws s3 ls s3://eurusd-ml-models/logs/ | sort | tail

# Download and read the latest log
aws s3 cp s3://eurusd-ml-models/logs/$(aws s3 ls s3://eurusd-ml-models/logs/ | sort | tail -n 1 | awk '{print $4}') - | less
```

### Option B: Check Instance State
The instance should be `stopped` most of the time. If it's `running` outside of the 22:10-22:30 UTC window, something might be stuck.

```bash
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=eurusd-retrain-worker" \
    --query "Reservations[0].Instances[0].{State:State.Name, LaunchTime:LaunchTime}"
```

### Option C: Live Monitoring (Dynamic SSH)
If you want to watch the process live (while the instance is running):

```bash
# 1. Get the dynamic IP and SSH in one command
# Note: This only works if the instance is RUNNING
ssh -i ~/.ssh/aws_keys/eurusd-keypair.pem ec2-user@$(aws ec2 describe-instances --filters "Name=tag:Name,Values=eurusd-retrain-worker" --query "Reservations[0].Instances[0].PublicDnsName" --output text) "tail -f /var/log/retrain.log"
```

*Tip: Add the `monitor_retrain` alias (see Section 4) to make this easier.*

## 3. Deployment / Redployment

Run these commands only during initial setup or if you need to fix a broken infrastructure.

### Prerequisites
- AWS CLI configured.
- `eurusd-keypair.pem` in `~/.ssh/aws_keys/`.

### Steps
1. **Deploy Network & EC2 Worker**:
   ```bash
   ./scripts/deployment/deploy_retrain_ec2.sh
   ```
   *This script is smart: it creates IAM roles, Security Groups, and the Instance if they don't exist. It also waits for the instance to self-initialize and shutdown.*

2. **Deploy Trigger Lambda**:
   ```bash
   ./scripts/deployment/deploy_lambda_trigger_retrain.sh
   ```
   *Updates the Lambda code and ensures the EventBridge schedule (22:10 UTC) is active.*

## 4. Helper Tools

### `scripts/mlops_utils/monitor_retrain.sh`
(Created for convenience)

```bash
# Check status
./scripts/mlops_utils/monitor_retrain.sh status

# See recent logs
./scripts/mlops_utils/monitor_retrain.sh logs

# Tail logs live (if running)
./scripts/mlops_utils/monitor_retrain.sh tail
```

## 5. Troubleshooting

**Q: The instance is stuck in "running" for hours.**
A: The script might have failed before the shutdown command. 
1. SSH in (using the command in Option C) and check `/var/log/retrain.log`.
2. Terminate the instance (`aws ec2 terminate-instances ...`) and redeploy using `./scripts/deployment/deploy_retrain_ec2.sh`. The new instance will auto-configure.

**Q: I don't see new logs in S3.**
A: Check the Lambda logs to see if it failed to start the instance.
```bash
aws logs tail /aws/lambda/eurusd-trigger-retrain --follow
```
