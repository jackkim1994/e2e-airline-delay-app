"""Lambda function to run filtering process"""
import json
import urllib.parse
import boto3
#from io import StringIO

print('Loading function')

s3 = boto3.client('s3')

def lambda_handler(event):
    """A Lambda handler to apply filtering into the raw data. 
       This python script will be used in Lambda AWS platform, not for Docker.

    Args:
        return the response content type 
        and re-direct the processed file into the destination bucket

    Raises:
        e: KeyError
        No return value. It will raise an exception alert if the key value does not match or not found.
    """
    print("Received event: " + json.dumps(event, indent=2))

    # Get the object from the event and show its content type
    bucket = event['Records'][0]['s3']['bucket']['name']
    destination_bucket = event['Records'][0]['s3']['bucket']['dest_name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    destination_key = key
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        print("CONTENT TYPE: " + response['ContentType'])

        s3.put_object(Bucket=destination_bucket, Key=key, Body=response['Body'].read())
        print(f"File {key} copied to {destination_key} ")
        return response['ContentType']

    except KeyError as e:
        print(e)
        print('Error getting object {} from bucket {}. \
              Make sure they exist and your bucket is \
              in the same region as this function.'.format(key, bucket))
        raise e
