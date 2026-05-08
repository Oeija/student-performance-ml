import json
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CONFIG
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = CONFIG.data_transformation

    def get_data_transformer_object(self):
        try:
            numerical_columns = self.data_transformation_config.numerical_columns
            categorical_columns = self.data_transformation_config.categorical_columns

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(
                f"Failed to create preprocessing pipeline: {e}", sys
            ) from e

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = self.data_transformation_config.target_column
            numerical_columns = self.data_transformation_config.numerical_columns

            # Validate target column exists
            if target_column_name not in train_df.columns:
                raise CustomException(
                    f"Target column '{target_column_name}' not found in training data", sys
                )

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(
                f"Train array shape: {train_arr.shape}, Test array shape: {test_arr.shape}"
            )

            # Extract and save feature names for interpretability
            try:
                feature_names = preprocessing_obj.get_feature_names_out()
                feature_names_list = list(feature_names)
                logging.info(f"Extracted {len(feature_names_list)} feature names")

                # Save feature names to JSON
                feature_names_path = self.data_transformation_config.feature_names_file_path
                with open(feature_names_path, "w") as f:
                    json.dump(feature_names_list, f, indent=2)
                logging.info(f"Saved feature names to {feature_names_path}")
            except Exception as e:
                logging.warning(f"Could not extract feature names: {e}")
                feature_names_list = None

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(
                f"Data transformation pipeline failed: {e}", sys
            ) from e
