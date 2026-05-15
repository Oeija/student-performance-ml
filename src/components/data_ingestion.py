import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.data_validation import DataValidation
from src.config import CONFIG
from src.exception import CustomException
from src.logger import logging


class DataIngestion:
    def __init__(self):
        self.ingestion_config = CONFIG.data_ingestion

    def _validate_file_exists(self) -> None:
        """Validate that the raw data file exists."""
        raw_path = self.ingestion_config.raw_data_path
        if not os.path.exists(raw_path):
            raise CustomException(
                f"Raw data file not found: {raw_path}", sys
            )
        logging.info(f"Raw data file validated: {raw_path}")

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
            data_validation = DataValidation()
            df = data_validation.validate_all(df)

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
