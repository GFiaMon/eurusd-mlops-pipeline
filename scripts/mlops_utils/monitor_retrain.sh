#!/bin/bash
#
# Helper script to monitor the EUR/USD Retraining EC2 Instance
#
# Usage:
#   ./monitor_retrain.sh status    # Check instance state
#   ./monitor_retrain.sh logs      # List S3 logs
#   ./monitor_retrain.sh tail      # Tail live logs via SSH (needs instance running)
#

KEY_PATH="~/.ssh/aws_keys/eurusd-keypair.pem"
TAG_NAME="eurusd-retrain-worker"
S3_BUCKET="eurusd-ml-models"

CMD=$1

if [ -z "$CMD" ]; then
    echo "Usage: $0 {status|logs|tail}"
    exit 1
fi

if [ "$CMD" == "status" ]; then
    echo "Checking instance status..."
    aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=$TAG_NAME" \
        --query "Reservations[0].Instances[0].{InstanceId:InstanceId, State:State.Name, LaunchTime:LaunchTime, PublicIpAddress:PublicIpAddress}" \
        --output table
elif [ "$CMD" == "logs" ]; then
    echo "Listing recent S3 logs..."
    aws s3 ls s3://$S3_BUCKET/logs/ | sort | tail -n 10
elif [ "$CMD" == "tail" ]; then
    echo "Finding instance DNS..."
    DNS=$(aws ec2 describe-instances --filters "Name=tag:Name,Values=$TAG_NAME" "Name=instance-state-name,Values=running" --query "Reservations[0].Instances[0].PublicDnsName" --output text)
    
    if [ "$DNS" == "None" ] || [ -z "$DNS" ]; then
        echo "Error: Instance is not running. Cannot tail logs."
        exit 1
    fi
    
    echo "Connecting to $DNS..."
    ssh -o StrictHostKeyChecking=no -i $KEY_PATH ec2-user@$DNS "tail -f /var/log/retrain.log"
else
    echo "Unknown command: $CMD"
    echo "Usage: $0 {status|logs|tail}"
    exit 1
fi
