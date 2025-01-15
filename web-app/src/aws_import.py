"""
Used to import and load models from S3 bucket
"""
import logging
import io

from typing import Union, Optional
import numpy as np
import pandas as pd
import boto3
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

logger = logging.getLogger("clouds")
def load_model_from_s3(bucket_name:str, object_key:str) -> Optional[Union[RandomForestRegressor, \
                                                                    GradientBoostingRegressor, \
                                                                    LinearRegression, \
                                                                    None]]:
    """
    Load a machine learning model from an AWS S3 bucket.
    Args:
        object_key (str): The key in the S3 bucket where the model is stored.
    Returns:
        The loaded model, or None if loading fails.
    """
    try:
        # Create an S3 client configured for the specified region.
        s3 = boto3.client('s3', region_name='us-east-2')
        # Retrieve the object from S3 by key.
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        # Read the binary content of the object.
        model_data = response['Body'].read()
        # Load the model from the binary data.
        model = joblib.load(io.BytesIO(model_data))
        logger.info(f'Model {object_key}-{model} successfully loaded from {bucket_name}')
        return model
    except ValueError as e:
        logger.error(f'Error loading model from S3: {e}')
        return None

def make_prediction(model:Optional[Union[RandomForestRegressor, \
                                         GradientBoostingRegressor, \
                                         LinearRegression]], \
                    input_data:pd.DataFrame) -> Optional[Union[np.ndarray,\
                                                               None]]:
    """
    Make predictions using the given model and input data.

    Args:
        model (object): The machine learning model to use for predictions.
        input_data (pd.DataFrame): The input data for making predictions.

    Returns:
        np.ndarray: The predictions made by the model.
    """
    try:
        res = model.predict(input_data)
        return res
    except ValueError as e:
        logger.error(f'Error making prediction: {e}')
        return None
