import os
import tempfile
import pandas as pd
import pytest

from src.components.data_ingestion import DataIngestion
from src.exception import CustomException


def create_mock_data(tmpdir):
    """Create a mock CSV file for testing."""
    data = {
        "gender": ["female", "male", "female"],
        "race_ethnicity": ["group A", "group B", "group C"],
        "parental_level_of_education": ["high school", "some college", "bachelor's degree"],
        "lunch": ["standard", "free/reduced", "standard"],
        "test_preparation_course": ["none", "completed", "none"],
        "math_score": [65, 72, 88],
        "reading_score": [70, 75, 90],
        "writing_score": [68, 74, 85],
    }
    df = pd.DataFrame(data)
    filepath = os.path.join(tmpdir, "test_students.csv")
    df.to_csv(filepath, index=False)
    return filepath


def test_data_ingestion_runs():
    """Test that data ingestion runs without crashing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_path = create_mock_data(tmpdir)
        
        ingestion = DataIngestion()
        # Temporarily override config path
        ingestion.ingestion_config.raw_data_path = mock_path
        
        train_path, test_path = ingestion.initiate_data_ingestion()
        
        assert os.path.exists(train_path)
        assert os.path.exists(test_path)
        assert os.path.exists(ingestion.ingestion_config.raw_data_file_path)


def test_validate_file_exists():
    """Test that missing file raises exception."""
    ingestion = DataIngestion()
    ingestion.ingestion_config.raw_data_path = "nonexistent_file.csv"
    
    with pytest.raises(CustomException):
        ingestion._validate_file_exists()
