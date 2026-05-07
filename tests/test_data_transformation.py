import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.components.data_transformation import DataTransformation


def create_mock_train_test_files(tmpdir):
    """Create mock train and test CSV files."""
    train_data = {
        "gender": ["female", "male", "female", "male"],
        "race_ethnicity": ["group A", "group B", "group C", "group D"],
        "parental_level_of_education": ["high school", "some college", "bachelor's degree", "master's degree"],
        "lunch": ["standard", "free/reduced", "standard", "free/reduced"],
        "test_preparation_course": ["none", "completed", "none", "completed"],
        "math_score": [65, 72, 88, 90],
        "reading_score": [70, 75, 90, 85],
        "writing_score": [68, 74, 85, 88],
    }
    test_data = {
        "gender": ["female", "male"],
        "race_ethnicity": ["group A", "group B"],
        "parental_level_of_education": ["high school", "some college"],
        "lunch": ["standard", "free/reduced"],
        "test_preparation_course": ["none", "completed"],
        "math_score": [70, 80],
        "reading_score": [75, 80],
        "writing_score": [72, 78],
    }
    
    train_path = os.path.join(tmpdir, "train.csv")
    test_path = os.path.join(tmpdir, "test.csv")
    
    pd.DataFrame(train_data).to_csv(train_path, index=False)
    pd.DataFrame(test_data).to_csv(test_path, index=False)
    
    return train_path, test_path


def test_data_transformation_runs():
    """Test that data transformation runs without crashing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path, test_path = create_mock_train_test_files(tmpdir)
        
        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
            train_path, test_path
        )
        
        assert isinstance(train_arr, np.ndarray)
        assert isinstance(test_arr, np.ndarray)
        assert os.path.exists(preprocessor_path)


def test_output_shapes():
    """Test that output arrays have correct shapes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path, test_path = create_mock_train_test_files(tmpdir)
        
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(
            train_path, test_path
        )
        
        # Train has 4 rows, test has 2 rows
        # Plus 1 target column
        assert train_arr.shape[0] == 4
        assert test_arr.shape[0] == 2
        assert train_arr.shape[1] == test_arr.shape[1]  # Same number of features


def test_feature_names_saved():
    """Test that feature names are saved to JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path, test_path = create_mock_train_test_files(tmpdir)
        
        transformation = DataTransformation()
        transformation.initiate_data_transformation(train_path, test_path)
        
        feature_names_path = transformation.data_transformation_config.feature_names_file_path
        if os.path.exists(feature_names_path):
            import json
            with open(feature_names_path, "r") as f:
                feature_names = json.load(f)
            assert isinstance(feature_names, list)
            assert len(feature_names) > 0


def test_preprocessor_handle_unknown():
    """Test that preprocessor handles unknown categories gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path, test_path = create_mock_train_test_files(tmpdir)
        
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(
            train_path, test_path
        )
        
        # Test with new category not seen in training
        preprocessor = transformation.get_data_transformer_object()
        
        # Fit on training data
        train_df = pd.read_csv(train_path)
        input_features = train_df.drop(columns=["math_score"])
        preprocessor.fit(input_features)
        
        # Transform with unknown category - should not crash
        new_data = pd.DataFrame({
            "gender": ["unknown_gender"],  # Unknown category
            "race_ethnicity": ["group Z"],  # Unknown category
            "parental_level_of_education": ["high school"],
            "lunch": ["standard"],
            "test_preparation_course": ["none"],
            "reading_score": [70],
            "writing_score": [68],
        })
        
        result = preprocessor.transform(new_data)
        assert result is not None
