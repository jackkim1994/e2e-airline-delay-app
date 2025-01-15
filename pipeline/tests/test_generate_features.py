"""
    Tests for generate_features function
    """
import pandas as pd
import pytest
import src.generate_features as gf

def test_generate_features_happy_path():
    """
    Tests for happy path
    """
    # Create a sample DataFrame with valid data
    data = pd.DataFrame({
        'Cancelled': [False, True],
        'DepTimeBlk': ['1000-1059', '1800-1859'],
        'Airline': ['Southwest Airlines Co.', 'United Air Lines Inc.'],
        'dept-type': ['large_airport', 'medium_airport'],
        'arr-type': ['medium_airport', 'large_airport']
    })
    # Expected DataFrame after processing
    expected_data = pd.DataFrame({
        'Cancelled': [False],
        'DepTimeBlk': ['1000-1059'],
        'Airline': ['Southwest Airlines Co.'],
        'dept-type': ['large_airport'],
        'arr-type': ['medium_airport'],
        'dep_time': [10],
        'Southwest Airlines Co.': [True],
        'dept-type_ohe_large_airport': [True],
        'arr-type_ohe_medium_airport': [True]
    })
    # Generate features using the function
    result_data = gf.generate_features(data)
    # Assert the results
    assert result_data.equals(expected_data)

def test_generate_features_unhappy_path():
    """
    Tests for value error
    """
    # Create a sample DataFrame with missing required columns
    data = pd.DataFrame({
        'Cancelled': [False, True, False],
        'DepTimeBlk': ['0000-00:59', '0100-0159', '0200-0259'],
        # 'Airline' column is missing
        'dept-type': ['large_airport', 'medium_airport', 'large_airport'],
        'arr-type': ['medium_airport', 'large_airport', 'medium_airport']
    })
    # The function is expected to raise a KeyError due to missing 'Airline' column
    with pytest.raises(KeyError):
        gf.generate_features(data)
