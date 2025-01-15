import logging
import io
import os
import sys
import boto3
import joblib
import pandas as pd
from botocore.stub import Stubber
from sklearn.linear_model import LogisticRegression
import pytest

# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, src_path)

import aws_import as ai

logger = logging.getLogger("clouds")

# Unit tests using pytest
class TestModelFunctions:

    @pytest.fixture
    def mock_model(self):
        model = LogisticRegression()
        model.fit([[0, 0], [1, 1]], [0, 1])
        return model

    @pytest.fixture
    def input_data(self):
        return pd.DataFrame([[1, 2], [3, 4]], columns=['feature1', 'feature2'])

    def test_load_model_from_s3_success(self, mock_model):
        s3 = boto3.client('s3')
        stubber = Stubber(s3)

        model_data = io.BytesIO()
        joblib.dump(mock_model, model_data)
        model_data.seek(0)

        response = {
            'Body': model_data
        }

        expected_params = {'Bucket': 'group4-final-project', 'Key': 'test-model.joblib'}
        stubber.add_response('get_object', response, expected_params)

        with stubber:
            loaded_model = ai.load_model_from_s3('test-bucket', 'test-model.joblib')
            assert loaded_model is not None
            assert hasattr(loaded_model, 'predict')

    def test_load_model_from_s3_failure(self):
        s3 = boto3.client('s3')
        stubber = Stubber(s3)
        
        stubber.add_client_error('get_object', 'NoSuchKey')

        with stubber:
            loaded_model = ai.load_model_from_s3('test-bucket', 'test-model.pkl')
            assert loaded_model is None

    def test_make_prediction_success(self, mock_model, input_data):
        predictions = ai.make_prediction(mock_model, input_data)
        assert predictions is not None
        assert len(predictions) == len(input_data)

    def test_make_prediction_failure(self, mock_model):
        predictions = ai.make_prediction(mock_model, pd.DataFrame([]))
        assert predictions is None

if __name__ == '__main__':
    pytest.main()
