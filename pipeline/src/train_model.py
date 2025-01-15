"""
This module trains the model
"""
import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("delay")

def train_test(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the input DataFrame into training and testing datasets.

    Parameters:
    - data (pd.DataFrame): The DataFrame to split.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: 
    A tuple containing the training dataset and the testing dataset.
    
    The function splits the data into 80% training and 20% testing subsets, 
    with a fixed random state for reproducibility.
    Logging is used to record the creation of the train-test split.
    """
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    logger.info("Train test split created")
    return train, test


def train_model_pcr(train: pd.DataFrame, config: Dict[str, Any]):
    '''
    Train model using PCR

    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        config (Dict[str, Any]): A dictionary containing the configuration parameters.

    Returns:
        Trained PCR model
    '''
    pcr_config = config.get("PCR", {})
    n_components = pcr_config.get("n_components", None)
    svd_solver = pcr_config.get("solver", "auto")
    x_train = train[config["features"]]
    y_train = train[config["response"]]

    scaler = StandardScaler()
    pca = PCA(n_components=n_components, svd_solver=svd_solver)
    linear_regression = LinearRegression()
    pcr = make_pipeline(scaler, pca, linear_regression)
    pcr.fit(x_train, y_train)
    logger.info("PCR model created")
    return pcr


def train_model_rf(train: pd.DataFrame, config: Dict[str, Any]):
    '''
    Train model using RF

    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        config (Dict[str, Any]): A dictionary containing the configuration parameters.

    Returns:
        Trained RF model
    '''
    rf_config = config.get("RF", {})
    n_estimators = rf_config.get("n_estimators", 100)
    max_depth = rf_config.get("max_depth", None)
    random_state = rf_config.get("random_state", None)
    x_train = train[config["features"]]
    y_train = train[config["response"]]

    if not isinstance(config["RF"]["n_estimators"], int) or n_estimators <= 0:
        raise ValueError("n_estimators must be a positive integer")
    if not isinstance(config["RF"]["max_depth"], int) or max_depth <= 0:
        raise ValueError("max_depth must be a positive integer")

    rf_model = RandomForestRegressor(n_estimators, max_depth, random_state)
    rf_model.fit(x_train, y_train)
    logger.info("Random Forest model created")
    return rf_model


def train_model_gbm(train: pd.DataFrame, config: Dict[str, Any]):
    '''
    Train model using GBM

    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        config (Dict[str, Any]): A dictionary containing the configuration parameters.

    Returns:
        Trained GBM model
    '''
    gbm_config = config.get("GBM", {})
    n_estimators = gbm_config.get("n_estimators", 100)
    learning_rate = gbm_config.get("learning_rate", 0.1)
    max_depth = gbm_config.get("max_depth", 3)
    random_state = gbm_config.get("random_state", None)
    x_train = train[config["features"]]
    y_train = train[config["response"]]

    if not isinstance(config["GBM"]["n_estimators"], int) or n_estimators <= 0:
        raise ValueError("n_estimators must be a positive integer")
    if not isinstance(config["GBM"]["max_depth"], int) or max_depth <= 0:
        raise ValueError("max_depth must be a positive integer")

    gbm = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                   max_depth=max_depth, random_state=random_state)
    gbm.fit(x_train, y_train)
    logger.info("GBM created")
    return gbm


def save_data(train: pd.DataFrame, test: pd.DataFrame, location: Path) -> None:
    '''
    Save train and test data as CSV files

    Args:
        train (pd.DataFrame): The DataFrame containing the training data.
        test (pd.DataFrame): The DataFrame containing the test data.
        location (Path): The directory location to save the CSV files.
    '''
    try:
        train.to_csv(location / "train.csv")
        test.to_csv(location / "test.csv")
        logger.info("Train data saved to %s", location / "train.csv")
        logger.info("Test data saved to %s", location / "test.csv")
    except FileNotFoundError as file_error:
        logger.error("File not found error occurred while saving data: %s", file_error)


def save_model(model, location: Path) -> None:
    '''
    Save trained model

    Args:
        model: The trained model.
        location (Path): The directory location to save the model file.
    '''
    try:
        joblib.dump(model, location)
        logger.info("Model saved to %s", location)
    except FileNotFoundError as file_error:
        logger.error("File not found error occurred while saving model: %s", file_error)
