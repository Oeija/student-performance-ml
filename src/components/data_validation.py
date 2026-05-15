import sys

import pandas as pd

from src.config import CONFIG
from src.exception import CustomException
from src.logger import logging


class DataValidation:
    def __init__(self):
        self.validation_config = CONFIG.validations
        self.transformation_config = CONFIG.data_transformation

    def _validate_no_nulls(self, df: pd.DataFrame) -> None:
        """Validate that no null values exist in the dataset."""
        null_counts = df.isna().sum()
        total_nulls = null_counts.sum()
        if total_nulls > 0:
            null_cols = null_counts[null_counts > 0].to_dict()
            raise CustomException(
                f"Dataset contains {total_nulls} null value(s) in columns: {null_cols}",
                sys,
            )
        logging.info("No null values found in dataset")

    def _validate_data_types(self, df: pd.DataFrame) -> None:
        """Validate that columns have correct data types."""
        score_columns = self.transformation_config.numerical_columns + [
            self.transformation_config.target_column
        ]

        for col in score_columns:
            if col not in df.columns:
                continue
            if df[col].dtype != "int64":
                raise CustomException(
                    f"Column '{col}' must be int64, got {df[col].dtype}. "
                    f"Score columns must contain whole numbers.",
                    sys,
                )

        categorical_columns = self.transformation_config.categorical_columns
        for col in categorical_columns:
            if col not in df.columns:
                continue
            if df[col].dtype != "object" and not pd.api.types.is_string_dtype(df[col]):
                raise CustomException(
                    f"Column '{col}' must be string/object, got {df[col].dtype}",
                    sys,
                )

        logging.info("Data types validated: scores are int64, categoricals are strings")

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

    def _validate_no_empty_strings(self, df: pd.DataFrame) -> None:
        """Validate that categorical columns have no empty, whitespace-only, or placeholder values."""
        categorical_columns = self.transformation_config.categorical_columns
        placeholders = {
            "N/A",
            "na",
            "unknown",
            "Unknown",
            "-",
            "?",
            "NULL",
            "null",
            "None",
            "",
        }

        for col in categorical_columns:
            if col not in df.columns:
                continue

            # Check for empty or whitespace-only values
            str_series = df[col].astype(str)
            empty_or_whitespace = str_series.str.strip() == ""
            if empty_or_whitespace.any():
                count = empty_or_whitespace.sum()
                raise CustomException(
                    f"Column '{col}' contains {count} empty or whitespace-only value(s)",
                    sys,
                )

            # Check for placeholder values
            mask = str_series.isin(placeholders)
            if mask.any():
                found_placeholders = sorted(str_series[mask].unique())
                raise CustomException(
                    f"Column '{col}' contains placeholder values: {found_placeholders}",
                    sys,
                )

        logging.info("No empty strings or placeholder values found")

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

    def validate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all validations and return cleaned DataFrame."""
        logging.info("Starting data validation")
        self._validate_columns(df)
        self._validate_no_nulls(df)
        self._validate_data_types(df)
        self._validate_score_ranges(df)
        self._validate_categorical_values(df)
        self._validate_no_empty_strings(df)
        df = self._remove_duplicates(df)
        logging.info("Data validation completed successfully")
        return df
