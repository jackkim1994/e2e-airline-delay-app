"""
This module evaluates model's performance
"""
from typing import Dict
from pathlib import Path

import logging
import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger("delay")

def evaluate_performance(scores: pd.DataFrame) -> Dict:
    '''
    Use scores dataframe to calculate mean_absolute_error, mean_squared_error, r2_score.

    Args:
        scores (Dict): A dictionary containing scores dataframe with keys "test", "prob", and "bin".

    Returns:
        Dict: A dictionary containing the calculated metrics.
    '''
    y_test = scores["test"]
    y_pred = scores["pred"]
    if len(y_test) <= 0:
        raise ValueError("Empty data")
    if np.isnan(y_test).any() or np.isnan(y_pred).any():
        raise ValueError("Input scores contain NaN values")

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_new = r2_score(y_test, y_pred)

    return {"mae": mae, "rmse": rmse, "r2": r2_new}


def save_metrics(metric: Dict, location: Path) -> None:
    '''
    Save metrics as yml file

    Args:
        metric (Dict[str, Any]): A dictionary containing the metrics to be saved.
        location (Path): The path to save the YAML file.
    '''
    try:
        if isinstance(metric.get("mae"), np.generic):
            mae_value = float(metric["mae"].item())
            metric["mae"] = mae_value
        if isinstance(metric.get("rmse"), np.generic):
            rmse_value = float(metric["rmse"].item())
            metric["rmse"] = rmse_value
        if isinstance(metric.get("r2"), np.generic):
            r2_value = float(metric["r2"].item())
            metric["r2"] = r2_value
        with open(location, "w", encoding="utf-8") as file:
            yaml.dump(metric, file)
        logger.info("Model metrics saved to %s", location)
    except FileNotFoundError as file_error:
        logger.error("File not found error occurred while saving dataset: %s", file_error)
