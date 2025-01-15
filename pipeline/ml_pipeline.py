import argparse
import datetime
import logging
import logging.config
import yaml
import pandas as pd

from pathlib import Path

import src.aws_utils as aws
import src.evaluate_performance as ep
import src.generate_features as gf
import src.score_model as sm
import src.train_model as tm

logging.config.fileConfig("config/logs/logging.conf")
logger = logging.getLogger("delay")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create features from flight delay data"
    )
    parser.add_argument(
        "--config", default="config/default-config.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()

    # Load configuration file for parameters and run config
    with open(args.config, "r") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.error.YAMLError as e:
            logger.error("Error while loading configuration from %s", args.config)
        else:
            logger.info("Configuration file loaded from %s", args.config)

    run_config = config.get("run_config", {})

    # Set up output directory for saving artifacts
    now = int(datetime.datetime.now().timestamp())
    artifacts = Path(run_config.get("output", "runs")) / str(now)
    artifacts.mkdir(parents=True)

    # Save config file to artifacts directory for traceability
    with (artifacts / "config.yaml").open("w") as f:
        yaml.dump(config, f)

    # Acquire data from repository and save to disk
    aws_config = config.get("aws")
    df = aws.read_csv_from_s3("flight_data_2021.csv", aws_config)
    # Remove this line for real data
    # df = df.drop(columns = "Unnamed: 0.1")
    
    # Clean datasets and generate features; save to disk
    data = gf.generate_features(df)
    print(data.columns)
    #gf.save_features(data, artifacts / "data.csv")

    # Split data into train/test set and train model based on config; save each to disk
    train, test = tm.train_test(data)
    #tm.save_data(train, test, artifacts)

    pcr = tm.train_model_pcr(train, config["train_model"])
    tm.save_model(pcr, artifacts / "pcr_model_object.joblib")

    rf = tm.train_model_rf(train, config["train_model"])
    tm.save_model(rf, artifacts / "rf_model_object.joblib")

    gbm = tm.train_model_gbm(train, config["train_model"])
    tm.save_model(gbm, artifacts / "gbm_model_object.joblib")

    # Score model on test set; save scores to disk
    scores_pcr = sm.score_model(test, pcr, config["score_model"])
    sm.save_scores(scores_pcr, artifacts / "scores_pcr.csv")

    scores_rf = sm.score_model(test, rf, config["score_model"])
    sm.save_scores(scores_rf, artifacts / "scores_rf.csv")

    scores_gbm = sm.score_model(test, gbm, config["score_model"])
    sm.save_scores(scores_gbm, artifacts / "scores_gbm.csv")

    # Evaluate model performance metrics; save metrics to disk
    metrics_pcr = ep.evaluate_performance(scores_pcr)
    ep.save_metrics(metrics_pcr, artifacts / "metrics_pcr.yaml")

    metrics_rf = ep.evaluate_performance(scores_rf)
    ep.save_metrics(metrics_rf, artifacts / "metrics_rf.yaml")

    metrics_gbm = ep.evaluate_performance(scores_gbm)
    ep.save_metrics(metrics_gbm, artifacts / "metrics_gbm.yaml")

    # Upload all artifacts to S3
    if aws_config["upload"] == True:
        aws.upload_artifacts(artifacts, aws_config)
        logger.info("File uploaded to S3")
    else:
        logger.info("Do not uploaded to S3")
