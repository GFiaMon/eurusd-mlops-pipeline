"""
AWS Lambda Function: Trigger EC2 Retraining Instance

This Lambda is triggered by EventBridge daily at 22:10 UTC (10 min after data ingestion).
It starts the dedicated EC2 instance that retrains the top 3 models.
"""

import json
import os
import boto3
from datetime import datetime

ec2 = boto3.client('ec2')

def lambda_handler(event, context):
    """
    Start the EC2 instance tagged for model retraining.
    
    Args:
        event: EventBridge event (scheduled trigger)
        context: Lambda context
        
    Returns:
        dict: Response with status and instance details
    """
    
    # Configuration
    retrain_ec2_tag = os.getenv('RETRAIN_EC2_TAG_NAME', 'eurusd-retrain-worker')
    
    print(f"[{datetime.utcnow().isoformat()}] Starting retraining workflow...")
    print(f"Looking for EC2 instance with tag: Name={retrain_ec2_tag}")
    
    try:
        # Find EC2 instance by tag
        response = ec2.describe_instances(
            Filters=[
                {'Name': 'tag:Name', 'Values': [retrain_ec2_tag]},
                {'Name': 'instance-state-name', 'Values': ['stopped', 'running']}
            ]
        )
        
        if not response['Reservations']:
            error_msg = f"No EC2 instance found with tag Name={retrain_ec2_tag}"
            print(f"ERROR: {error_msg}")
            return {
                'statusCode': 404,
                'body': json.dumps({'error': error_msg})
            }
        
        instance_id = response['Reservations'][0]['Instances'][0]['InstanceId']
        instance_state = response['Reservations'][0]['Instances'][0]['State']['Name']
        
        print(f"Found instance: {instance_id} (state: {instance_state})")
        
        # Start instance if stopped
        if instance_state == 'stopped':
            print(f"Starting instance {instance_id}...")
            ec2.start_instances(InstanceIds=[instance_id])
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Retraining EC2 instance started successfully',
                    'instance_id': instance_id,
                    'action': 'started',
                    'timestamp': datetime.utcnow().isoformat()
                })
            }
        
        elif instance_state == 'running':
            print(f"Instance {instance_id} is already running")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Retraining EC2 instance already running',
                    'instance_id': instance_id,
                    'action': 'already_running',
                    'timestamp': datetime.utcnow().isoformat()
                })
            }
        
        else:
            warning_msg = f"Instance {instance_id} is in unexpected state: {instance_state}"
            print(f"WARNING: {warning_msg}")
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'warning': warning_msg,
                    'instance_id': instance_id,
                    'state': instance_state
                })
            }
            
    except Exception as e:
        error_msg = f"Failed to start retraining EC2: {str(e)}"
        print(f"ERROR: {error_msg}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': error_msg})
        }
