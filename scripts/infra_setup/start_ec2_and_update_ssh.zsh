#!/bin/zsh

# Configuration
TAG_NAME="eurusd-predictor"
SSH_HOST_ALIAS="capst2-inference-app"  # Change this to match your SSH config Host entry

echo "Starting EC2 instance with tag Name:${TAG_NAME}..."

# Find instance ID by tag
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=${TAG_NAME}" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

if [ "$INSTANCE_ID" = "None" ] || [ -z "$INSTANCE_ID" ]; then
  echo "Error: No instance found with tag Name:${TAG_NAME}"
  exit 1
fi

echo "Found instance: ${INSTANCE_ID}"

# Start the instance
aws ec2 start-instances --instance-ids "$INSTANCE_ID" > /dev/null

echo "Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

# Get the public DNS name
NEW_PUBLIC_DNS=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicDnsName' \
  --output text)

if [ -z "$NEW_PUBLIC_DNS" ] || [ "$NEW_PUBLIC_DNS" = "None" ]; then
  echo "Error: Could not retrieve public DNS name"
  exit 1
fi

echo "Public DNS: ${NEW_PUBLIC_DNS}"

# Update SSH config (macOS-compatible sed)
if [ -f ~/.ssh/config ]; then
  # Create backup
  cp ~/.ssh/config ~/.ssh/config.bak
  
  # Update the HostName line for the specified host (case-insensitive)
  sed -i '' "/Host ${SSH_HOST_ALIAS}/,/^Host /{
    s/^\([[:space:]]*[Hh]ost[Nn]ame[[:space:]]\).*/\1${NEW_PUBLIC_DNS}/
  }" ~/.ssh/config
  
  echo "✓ SSH config updated for host '${SSH_HOST_ALIAS}'"
  echo "  HostName: ${NEW_PUBLIC_DNS}"
else
  echo "Warning: ~/.ssh/config not found"
fi

echo "Done!"
