import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.config import CONFIG


class DataIngestion:
    def __init__(self):
        self.ingestion_config = CONFIG.data_ingestion
        self.validation_config = CONFIG.validations
        self.transformation_config = CONFIG.data_transformation

    def _validate_file_exists(self) -> None:
        """Validate that the raw data file exists."""
        raw_path = self.ingestion_config.raw_data_path
        if not os.path.exists(raw_path):
            raise CustomException(
                f"Raw data file not found: {raw_path}", sys
            )
        logging.info(f"Raw data file validated: {raw_path}")

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that all expected columns are present."""
        expected_columns = (
            self.transformation_config.numerical_columns
            + self.transformation_config.categorical_columns
            + [self.transformation_config.target_column]
        )
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            raise CustomException(
                f"Missing columns in dataset: {missing_columns}", sys
            )
        logging.info(f"All {len(expected_columns)} expected columns validated")

    def _validate_score_ranges(self, df: pd.DataFrame) -> None:
        """Validate that score columns are within valid range."""
        score_columns = self.transformation_config.numerical_columns + [
            self.transformation_config.target_column
        ]
        for col in score_columns:
            if not df[col].between(
                self.validation_config.score_min, self.validation_config.score_max
            ).all():
                raise CustomException(
                    f"Column '{col}' contains values outside range "
                    f"[{self.validation_config.score_min}, {self.validation_config.score_max}]",
                    sys,
                )
        logging.info(
            f"Score ranges validated: [{self.validation_config.score_min}, "
            f"{self.validation_config.score_max}]"
        )

    def _validate_categorical_values(self, df: pd.DataFrame) -> None:
        """Validate that categorical columns contain only expected values."""
        categorical_validations = {
            "gender": self.validation_config.valid_genders,
            "race_ethnicity": self.validation_config.valid_race_ethnicity,
            "parental_level_of_education": self.validation_config.valid_parental_education,
            "lunch": self.validation_config.valid_lunch,
            "test_preparation_course": self.validation_config.valid_test_preparation,
        }

        for col, valid_values in categorical_validations.items():
            invalid_values = set(df[col].unique()) - set(valid_values)
            if invalid_values:
                raise CustomException(
                    f"Column '{col}' contains invalid values: {invalid_values}. "
                    f"Expected: {valid_values}",
                    sys,
                )
        logging.info("Categorical values validated")

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows and log the count."""
        initial_count = len(df)
        df = df.drop_duplicates()
        removed_count = initial_count - len(df)
        if removed_count > 0:
            logging.warning(f"Removed {removed_count} duplicate rows")
        else:
            logging.info("No duplicate rows found")
        return df

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Validate raw data file exists
            self._validate_file_exists()

            # Read dataset
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info(f"Read the dataset as dataframe with shape {df.shape}")

            # Standardize column names
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_")

            # Validate data quality
            self._validate_columns(df)
            self._validate_score_ranges(df)
            self._validate_categorical_values(df)
            df = self._remove_duplicates(df)

            # Create processed and artifacts directories
            os.makedirs(self.ingestion_config.processed_data_dir, exist_ok=True)
            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_file_path, index=False, header=True)
            logging.info(f"Saved raw data to {self.ingestion_config.raw_data_file_path}")

            # Train-test split
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(
                df,
                test_size=self.ingestion_config.test_size,
                random_state=self.ingestion_config.random_state,
            )
            logging.info(
                f"Split data into train ({len(train_set)} rows) and "
                f"test ({len(test_set)} rows)"
            )

            # Save split data
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info("Saved train and test datasets")

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(
                f"Data ingestion pipeline failed: {e}", sys
            ) from e


if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation
    from src.components.model_trainer import ModelTrainer

    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
