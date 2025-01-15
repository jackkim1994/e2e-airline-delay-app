"""
This module sets up aws
"""
import logging
from pathlib import Path
from typing import Dict, List
from io import StringIO

import boto3
import pandas as pd

logger = logging.getLogger("delay")

def read_csv_from_s3(file_key: str, config: Dict) -> pd.DataFrame:
    """Read a CSV file from S3 into a Pandas DataFrame."""

    s3_session = boto3.client('s3')
    bucket_name = config["bucket_name"]
    # Get object from S3
    obj = s3_session.get_object(Bucket=bucket_name, Key=file_key)
    df_new = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
    return df_new

def upload_artifacts(artifacts: Path, config: Dict) -> List[str]:
    """Upload all the artifacts in the specified directory to S3.

    Args:
        artifacts (Path): 
        Directory containing all the artifacts from a given experiment.
        config (Dict): Config required to upload artifacts to S3
    Returns:
        List[str]: List of S3 URIs for each file that was uploaded.
    """

    # Initialize S3 client
    s3_client = boto3.client('s3')

    # List of uploaded file paths
    uploaded_files = []
    bucket_name = config["bucket_name"]
    s3_directory = "model-artifacts"

    # Iterate over files in the directory
    for file_path in artifacts.glob("*"):
        if file_path.is_file():
            # Construct S3 key (object key)
            s3_key = f"{s3_directory}/{file_path.name}"
            try:
                # Upload file to S3
                s3_client.upload_file(str(file_path), bucket_name, s3_key)

                # Append S3 URI to the list
                uploaded_files.append(f"s3://{bucket_name}/{s3_key}")

            except FileNotFoundError as file_error:
                logger.error("File not found error %s to S3: %s", file_path, file_error)

    logger.info("File uploaded to S3")
    return uploaded_files
