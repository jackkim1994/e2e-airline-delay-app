"""
Construct web application using streamlit
"""
# Import packages
import argparse
import logging.config
import streamlit as st
import pandas as pd
import yaml
from pathlib import Path

import aws_import as ai

logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("clouds")

# Parse command line arguments
parser = argparse.ArgumentParser(
description="Acquire, clean, and create features from clouds data"
)
parser.add_argument(
"--config", default="config/config.yaml", help="Path to configuration file"
)
args = parser.parse_args()

# Load configuration file for parameters and run config
with open(args.config, "r", encoding="utf-8") as f:
    try:
        config = yaml.load(f, Loader=yaml.FullLoader)
    except yaml.error.YAMLError as e:
        logger.error("Error while loading configuration from %s: %s", args.config, e)
    else:
        logger.info("Configuration file loaded from %s", args.config)

if config:
    # Set the config for AWS
    aws_config = config["aws"]

    ######################
    #### Streamlit app ###
    ######################

    st.set_page_config(
    page_title="Under which circumstances can the flight be delayed?",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        :root {
            --primary-color: #4A90E2;
            --background-light: #f0f2f6;
            --background-dark: #1e1e1e;
            --text-light: #000;
            --text-dark: #fff;
            --button-bg-light: #4A90E2;
            --button-bg-dark: #357ABD;
            --button-text-light: white;
            --button-text-dark: white;
        }

        @media (prefers-color-scheme: dark) {
            .main {
                background-color: var(--background-dark);
                color: var(--text-dark);
            }
            h1 {
                color: var(--primary-color);
            }
            .stButton>button {
                color: var(--button-text-dark);
                background: var(--button-bg-dark);
                border-radius: 8px;
                padding: 10px 24px;
                border: none;
                cursor: pointer;
                font-size: 16px;
            }
            .stButton>button:hover {
                background: var(--button-bg-dark);
            }
        }

        @media (prefers-color-scheme: light) {
            .main {
                background-color: var(--background-light);
                color: var(--text-light);
            }
            h1 {
                color: var(--primary-color);
            }
            .stButton>button {
                color: var(--button-text-light);
                background: var(--button-bg-light);
                border-radius: 8px;
                padding: 10px 24px;
                border: none;
                cursor: pointer;
                font-size: 16px;
            }
            .stButton>button:hover {
                background: var(--button-bg-light);
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Define the features and their types
    numerical_features = [
        "dept-elevation_ft", "arr-elevation_ft", "Route_Popularity", "Distance_Final",
        "Wind_Speed_mph", "Wind_Gust_mph", "Visibility_miles", "tempF", "precip_in",
        "daily_snow_in", "dep_time"
    ]

    categorical_features = [
        "Airline", "Departure Type", "Arrival Type"
    ]

    # Define the default values for numerical features
    default_numerical_values = {
        "dept-elevation_ft": 107.00, "arr-elevation_ft": 607.00, 
        "Route_Popularity": 2526.00, "Distance_Final": 754.00,
        "Wind_Speed_mph": 9.00, "Wind_Gust_mph": 14.00, "Visibility_miles": 3, 
        "tempF": 46.00, "precip_in": 0.00,
        "daily_snow_in": 0.00, "dep_time": 900.00
    }

    # Define the options for categorical features
    default_categorical_options = {
        "Airline": ["Air Wisconsin Airlines Corp", "Alaska Airlines Inc.", "Allegiant Air",
                     "American Airlines Inc.",
                    "Capital Cargo International", "Comair Inc.", 
                    "Commutair Aka Champlain Enterprises, Inc.",
                    "Delta Air Lines Inc.", "Endeavor Air Inc.", "Envoy Air", 
                    "Frontier Airlines Inc.",
                    "GoJet Airlines, LLC d/b/a United Express", "Horizon Air", 
                    "JetBlue Airways", "Mesa Airlines Inc.",
                    "Republic Airlines", "SkyWest Airlines Inc.", "Southwest Airlines Co.", 
                    "Spirit Air Lines",
                    "United Air Lines Inc."],
         "Departure Type": ["Closed", "Large Airport", "Medium Airport", "Small Airport"],
        "Arrival Type": ["Closed", "Large Airport", "Medium Airport", "Small Airport"]
    }
    # Map user-friendly names to the original feature names
    categorical_feature_mapping = {
        "Closed": {"Departure Type": "dept-type_ohe_closed", 
                   "Arrival Type": "arr-type_ohe_closed"},
        "Large Airport": {"Departure Type": "dept-type_ohe_large_airport",
                           "Arrival Type": "arr-type_ohe_large_airport"},
        "Medium Airport": {"Departure Type": "dept-type_ohe_medium_airport",
                            "Arrival Type": "arr-type_ohe_medium_airport"},
        "Small Airport": {"Departure Type": "dept-type_ohe_small_airport",
                           "Arrival Type": "arr-type_ohe_small_airport"},
    }

    # Streamlit interface
    st.title("Flight Delay Prediction")

    # Model selection dropdown
    st.write("#### Select Model Version")
    model_selection = st.selectbox("Select Model", ["PCR", "Random Forest",
                                                     "Gradient Boosting"])

    @st.cache_data
    def load_model(bucket_name, object_key):
        return ai.load_model_from_s3(bucket_name, object_key)

    def clear_cache():
        st.cache_data.clear()

    if st.button("Load Model"):
        # Clear cache if the model is changed
        if "model_version" in st.session_state and st.session_state["model_version"] != model_selection:
            clear_cache()
        if model_selection == "PCR":
            object_key = aws_config['pcr_key']
        elif model_selection == "Random Forest":
            object_key = aws_config['rf_key']
        elif model_selection == "Gradient Boosting":
            object_key = aws_config['gb_key']
        # Set the bucket name and model key for s3 load
        try:
            imported_model = load_model(aws_config["bucket_name"], object_key)
            if imported_model:
                st.session_state["model"] = imported_model
                st.session_state["model_version"] = model_selection
                st.success(f"{model_selection} loaded successfully!")
                logger.info(f"{model_selection} loaded successfully!")
            else:
                st.session_state["model"] = None
                st.error("Model loaded is None.")
        except ValueError as value_error:
            st.error(f"Failed to load {model_selection}: {value_error}")
            logger.error(f"Failed to load {model_selection}: {value_error}") 
    # Set Text Input title
    st.write("Enter Flight Details:")  
    # Create inputs for numerical features
    numerical_inputs = {}
    for feature in numerical_features:
        numerical_inputs[feature] = st.text_input(
            feature,
            value=default_numerical_values[feature]
        ) 
    # Date input for extracting Quarter, Month, and DayOfWeek
    date_input = st.date_input("Select Date", 
        value=st.session_state.get("date_input", pd.to_datetime("2021-01-01")),
        on_change=clear_cache(),
        args=("date_input", st.session_state.get("date_input", pd.to_datetime("2021-01-01"))))

    # Extract Quarter, Month, and DayOfWeek from the date input
    # These will combine into categorical features
    quarter = (date_input.month - 1) // 3 + 1
    month = date_input.month
    day_of_week = date_input.weekday() + 1

    # Create dropdowns for categorical features
    categorical_inputs = {}
    for feature in categorical_features:
        categorical_inputs[feature] = st.selectbox(
            feature,
            options=default_categorical_options[feature],
            index=default_categorical_options[feature].index(st.session_state.get(feature, default_categorical_options[feature][0])),
            on_change=clear_cache(),
            args=(feature, st.session_state.get(feature, default_categorical_options[feature][0]))
        ) 
    # Add extracted date-related features to inputs
    numerical_inputs["Quarter"] = quarter
    numerical_inputs["Month"] = month
    numerical_inputs["DayOfWeek"] = day_of_week

    # If the raw data requires one-hot encoding again before feeding into the model
    encoded_inputs = {}
    for feature in default_categorical_options.keys():
        if feature in ["Airline", "Departure Type", "Arrival Type"]:
            if feature == "Airline":
                for option in default_categorical_options[feature]:
                    encoded_inputs[f"{option}"] = int(categorical_inputs[feature] == option)
            else:
                for option, original_feature in categorical_feature_mapping.items():
                    encoded_inputs[original_feature[feature]] = int(categorical_inputs[feature] == option)


    # Reorder numerical inputs to match the order of the feature names
    num_order = config["column_order"]["num_features"]
    numerical_inputs = {key: numerical_inputs[key] for key in num_order}
    # Combine the numerical and categorical inputs
    input_features = {**numerical_inputs, **encoded_inputs}
    input_features = pd.DataFrame(input_features,index=[0])

    # Dataset
    st.write(input_features)
    # Feature engineering calculations
    if st.button("Make Prediction"):
        if "model" in st.session_state:
            try:        
                # Clear the cache to ensure the latest prediction is shown
                clear_cache()     
                # Make predictions
                pred = ai.make_prediction(st.session_state["model"], input_features)
                st.write(f"### Expected Delays: {pred} minutes")
                logger.info("Prediction completed.")
            except ValueError as e:
                st.error(f"Error making prediction: {e}")
                logger.error("Error making prediction e %s:", e)
        else:
            st.error("Please load a model first.")
else:
    st.error("Failed to load configuration.")
