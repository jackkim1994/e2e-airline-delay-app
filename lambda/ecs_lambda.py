"""Lambda function to run trigger ECR/ECS Pipeline"""
import json
import boto3

print('Loading function')

s3 = boto3.client('ecs')

def lambda_handler():
    """A Lambda handler to trigger ECR/ECS Pipeline
       This python script will be used in Lambda AWS platform, not for Docker.
       Note that this handler does not have Args or Return Value.
    
    """
    client = boto3.client('ecs')
    response = client.run_task(
        cluster='cloud-project-pipeline',
        launchType='FARGATE',
        taskDefinition='cloud-project-pipeline-v2',
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': [
                    'subnet-0169d9e6343f8a1dd',  # Replace with your subnet ID
                    'subnet-07e50de48236c1c32',
                    'subnet-09820407b79b4673f'
                ],
                'assignPublicIp': 'ENABLED'  # 'DISABLED' if you don't need a public IP
            }
        }
    )
    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }
