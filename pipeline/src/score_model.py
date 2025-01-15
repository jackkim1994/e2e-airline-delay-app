"""
This module save scores of models
"""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("delay")

def score_model(test: pd.DataFrame, tmo, config: dict) -> pd.DataFrame:
    """
    Score the model on the test data.

    Args:
        test (pd.DataFrame): The test dataset.
        tmo (RandomForestClassifier): The trained model object.
        config (dict): A dictionary containing configuration parameters.

    Returns:
        pd.DataFrame: A DataFrame containing the scores.
    """
    ypred = tmo.predict(test[config["features"]])
    scores = pd.DataFrame({"test": test[config["response"]], "pred": ypred})
    return scores


def save_scores(scores: pd.DataFrame, location: Path) -> None:
    '''
    Save scores DataFrame to a CSV file.

    Args:
        scores (pd.DataFrame): The DataFrame containing scores.
        location (Path): The file path to save the scores CSV.
    '''
    try:
        scores.to_csv(location)
        logger.info("Model scores saved to %s", location)
    except FileNotFoundError as file_error:
        logger.error("File not found error occurred while saving scores: %s", file_error)
        