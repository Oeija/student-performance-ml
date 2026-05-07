import pytest
from src.config import load_config, ProjectConfig


def test_load_config():
    """Test that config loads successfully."""
    config = load_config()
    assert isinstance(config, ProjectConfig)


def test_config_paths():
    """Test that config paths are valid strings."""
    config = load_config()
    assert config.data_ingestion.raw_data_path == "data/raw/StudentsPerformance.csv"
    assert config.data_transformation.target_column == "math_score"


def test_config_validation_ranges():
    """Test that validation ranges make sense."""
    config = load_config()
    assert config.validations.score_min == 0
    assert config.validations.score_max == 100
    assert config.model_trainer.cv_folds >= 2
    assert 0 < config.data_ingestion.test_size < 1


def test_config_invalid_test_size():
    """Test that invalid test size raises validation error."""
    from pydantic import ValidationError

    config_dict = {
        "data_ingestion": {
            "raw_data_path": "test.csv",
            "artifacts_dir": "artifacts",
            "train_filename": "train.csv",
            "test_filename": "test.csv",
            "raw_filename": "data.csv",
            "test_size": 1.5,  # Invalid: > 1
            "random_state": 42,
        },
        "data_transformation": {
            "target_column": "math_score",
            "numerical_columns": ["reading_score"],
            "categorical_columns": ["gender"],
            "preprocessor_filename": "preprocessor.pkl",
            "feature_names_filename": "feature_names.json",
        },
        "model_trainer": {
            "model_filename": "model.pkl",
            "cv_folds": 5,
            "min_score_threshold": 0.6,
        },
        "model_evaluation": {
            "shap_summary_plot": "shap_summary.png",
            "shap_importance_csv": "shap_importance.csv",
        },
        "validations": {
            "score_min": 0,
            "score_max": 100,
            "valid_genders": ["female", "male"],
            "valid_race_ethnicity": ["group A"],
            "valid_parental_education": ["high school"],
            "valid_lunch": ["standard"],
            "valid_test_preparation": ["none"],
        },
    }

    with pytest.raises(ValidationError):
        ProjectConfig(**config_dict)
