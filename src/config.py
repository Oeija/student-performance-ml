import os
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field, field_validator


class DataIngestionConfig(BaseModel):
    raw_data_path: str
    processed_data_dir: str
    artifacts_dir: str
    train_filename: str
    test_filename: str
    raw_filename: str
    test_size: float = Field(gt=0, lt=1)
    random_state: int

    @property
    def train_data_path(self) -> str:
        return os.path.join(self.processed_data_dir, self.train_filename)

    @property
    def test_data_path(self) -> str:
        return os.path.join(self.processed_data_dir, self.test_filename)

    @property
    def raw_data_file_path(self) -> str:
        return os.path.join(self.artifacts_dir, self.raw_filename)


class DataTransformationConfig(BaseModel):
    target_column: str
    numerical_columns: List[str]
    categorical_columns: List[str]
    preprocessor_filename: str
    feature_names_filename: str

    @property
    def preprocessor_obj_file_path(self) -> str:
        return os.path.join("artifacts", self.preprocessor_filename)

    @property
    def feature_names_file_path(self) -> str:
        return os.path.join("artifacts", self.feature_names_filename)


class ModelTrainerConfig(BaseModel):
    model_filename: str
    cv_folds: int = Field(ge=2)
    min_score_threshold: float

    @property
    def trained_model_file_path(self) -> str:
        return os.path.join("artifacts", self.model_filename)


class ModelEvaluationConfig(BaseModel):
    shap_summary_plot: str
    shap_importance_csv: str

    @property
    def shap_summary_plot_path(self) -> str:
        return os.path.join("artifacts", self.shap_summary_plot)

    @property
    def shap_importance_csv_path(self) -> str:
        return os.path.join("artifacts", self.shap_importance_csv)


class ValidationConfig(BaseModel):
    score_min: int = Field(ge=0)
    score_max: int = Field(le=100)
    valid_genders: List[str]
    valid_race_ethnicity: List[str]
    valid_parental_education: List[str]
    valid_lunch: List[str]
    valid_test_preparation: List[str]


class ProjectConfig(BaseModel):
    data_ingestion: DataIngestionConfig
    data_transformation: DataTransformationConfig
    model_trainer: ModelTrainerConfig
    model_evaluation: ModelEvaluationConfig
    validations: ValidationConfig


def load_config(config_path: str = "config/config.yaml") -> ProjectConfig:
    """Load and validate configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return ProjectConfig(**config_dict)


# Singleton instance for global access
CONFIG = load_config()
