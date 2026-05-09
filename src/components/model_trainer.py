import json
import os
import sys

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.config import CONFIG
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object
from src.components.model_interpretability import ModelInterpretability


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = CONFIG.model_trainer

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Ridge": Ridge(),
                "K-Neighbour Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(
                    verbose=False, allow_writing_files=False
                ),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Ridge": {
                    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                    "fit_intercept": [True, False],
                },
                "K-Neighbour Regressor": {
                    "n_neighbors": [5, 7, 9, 11],
                },
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            # Print full model comparison
            print("\n========== Model Performance Report ==========")
            for model_name, score in model_report.items():
                print(f"{model_name}: {score:.6f}")
            print("==============================================\n")

            # To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            print(f"Best Model: {best_model_name} | Score: {best_model_score:.6f}\n")
            logging.info(f"Best Model: {best_model_name} | Score: {best_model_score}")

            best_model = models[best_model_name]

            if best_model_score < self.model_trainer_config.min_score_threshold:
                raise CustomException(
                    f"No best model found. Best score {best_model_score} is below "
                    f"threshold {self.model_trainer_config.min_score_threshold}",
                    sys,
                )
            logging.info(f"Best found model on both training and testing dataset")

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            logging.info(f"Saved best model: {best_model_name}")

            # SHAP Interpretability
            try:
                # Load feature names if available
                feature_names = None
                feature_names_path = CONFIG.data_transformation.feature_names_file_path
                if os.path.exists(feature_names_path):
                    with open(feature_names_path, "r") as f:
                        feature_names = json.load(f)
                    logging.info(f"Loaded {len(feature_names)} feature names for SHAP")

                interpretability = ModelInterpretability()
                interpretability.explain_model(best_model, X_train, feature_names)
            except Exception as e:
                logging.warning(
                    f"SHAP interpretability analysis skipped due to error: {e}"
                )

            # Final evaluation
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(
                f"Model training pipeline failed: {e}", sys
            ) from e
