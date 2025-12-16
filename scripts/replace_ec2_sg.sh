#!/bin/bash

# --- CONFIGURATION VARIABLES ---
NEW_SG_NAME="eurusd-api-sg"
OLD_SG_NAME="launch-wizard-2"

# --- 1. GET INSTANCE ID ---
# Find the ID of your running EC2 instance (assuming you have only one running free-tier instance)
echo "Finding the running EC2 Instance ID..."
INSTANCE_ID=$(aws ec2 describe-instances \
    --filters Name=instance-state-name,Values=running \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text)

if [ -z "$INSTANCE_ID" ]; then
    echo "ERROR: Could not find a running EC2 instance. Please check AWS Console."
    exit 1
fi
echo "Found Instance ID: $INSTANCE_ID"


# --- 2. GET SECURITY GROUP IDs ---
# Get the ID of the new security group (using the name you created)
echo "Getting ID for new Security Group: $NEW_SG_NAME"
NEW_SG_ID=$(aws ec2 describe-security-groups \
    --group-names "$NEW_SG_NAME" \
    --query 'SecurityGroups[0].GroupId' \
    --output text)

if [ -z "$NEW_SG_ID" ]; then
    echo "ERROR: Could not find the new Security Group ID. Did you run the creation script?"
    exit 1
fi
echo "New Security Group ID: $NEW_SG_ID"


# --- 3. ATTACH NEW AND DETACH OLD GROUP ---
# The --groups parameter takes a LIST of ALL desired Security Group IDs. 
# We only want the new one, so we only pass the NEW_SG_ID.
echo "Attaching $NEW_SG_ID and detaching all others..."
aws ec2 modify-instance-attribute \
    --instance-id "$INSTANCE_ID" \
    --groups "$NEW_SG_ID"

echo "✅ Success: Instance $INSTANCE_ID now only has Security Group $NEW_SG_NAME attached."


# --- 4. DELETE OLD SECURITY GROUP ---
# Get the ID of the old security group to delete it later
echo "Getting ID for old Security Group: $OLD_SG_NAME"
OLD_SG_ID=$(aws ec2 describe-security-groups \
    --group-names "$OLD_SG_NAME" \
    --query 'SecurityGroups[0].GroupId' \
    --output text)

# The old group is no longer associated, so we can delete it.
if [ ! -z "$OLD_SG_ID" ] && [ "$OLD_SG_ID" != "None" ]; then
    echo "Deleting old Security Group: $OLD_SG_ID ($OLD_SG_NAME)..."
    aws ec2 delete-security-group \
        --group-id "$OLD_SG_ID"
    echo "✅ Success: Old Security Group deleted."
else
    echo "Old Security Group not found or already deleted. Skipping deletion."
fi