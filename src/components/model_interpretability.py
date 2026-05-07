import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Use non-interactive backend

from src.config import CONFIG
from src.exception import CustomException
from src.logger import logging

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not installed. Model interpretability will be skipped.")


class ModelInterpretability:
    def __init__(self):
        self.evaluation_config = CONFIG.model_evaluation

    def explain_model(self, model, X_train, feature_names):
        """Generate SHAP explanations for the trained model.

        Args:
            model: Trained sklearn model
            X_train: Training features (numpy array)
            feature_names: List of feature names
        """
        if not SHAP_AVAILABLE:
            logging.info("SHAP not available, skipping interpretability")
            return

        try:
            logging.info("Starting SHAP model interpretability analysis")

            # Ensure X_train is a numpy array
            if hasattr(X_train, "toarray"):
                X_train = X_train.toarray()

            # Sample data for SHAP (can be slow on full dataset)
            sample_size = min(500, len(X_train))
            X_sample = shap.sample(X_train, sample_size, random_state=42)

            # Choose explainer based on model type
            model_name = type(model).__name__
            logging.info(f"Using SHAP explainer for model: {model_name}")

            if hasattr(model, "coef_"):
                # Linear models (Ridge, LinearRegression)
                explainer = shap.LinearExplainer(model, X_sample)
            elif hasattr(model, "feature_importances_"):
                # Tree-based models
                explainer = shap.TreeExplainer(model)
            else:
                # Fallback to KernelExplainer
                explainer = shap.KernelExplainer(model.predict, X_sample)

            shap_values = explainer.shap_values(X_sample)

            # Handle binary classification case
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class

            # 1. Generate SHAP summary plot (global interpretability)
            self._generate_summary_plot(shap_values, X_sample, feature_names)

            # 2. Generate feature importance CSV
            self._generate_importance_csv(shap_values, feature_names)

            logging.info("SHAP interpretability analysis completed")

        except Exception as e:
            logging.error(f"SHAP model interpretability analysis failed: {e}")
            # Don't raise exception - interpretability is optional

    def _generate_summary_plot(self, shap_values, X_sample, feature_names):
        """Generate and save SHAP summary plot."""
        try:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                show=False,
                plot_size=(10, 6),
            )
            plot_path = self.evaluation_config.shap_summary_plot_path
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            logging.info(f"Saved SHAP summary plot to {plot_path}")
        except Exception as e:
            logging.error(f"Failed to generate SHAP summary plot: {e}")

    def _generate_importance_csv(self, shap_values, feature_names):
        """Generate and save feature importance CSV."""
        try:
            importance = pd.DataFrame(
                {
                    "feature": feature_names,
                    "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                }
            ).sort_values("mean_abs_shap", ascending=False)

            csv_path = self.evaluation_config.shap_importance_csv_path
            importance.to_csv(csv_path, index=False)
            logging.info(f"Saved SHAP feature importance to {csv_path}")

            # Log top 5 features
            top_features = importance.head(5)["feature"].tolist()
            logging.info(f"Top 5 most important features: {top_features}")
        except Exception as e:
            logging.error(f"Failed to generate SHAP importance CSV: {e}")
