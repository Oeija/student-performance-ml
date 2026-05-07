import numpy as np
import pytest

from src.components.model_trainer import ModelTrainer
from src.config import CONFIG


def create_mock_train_test_arrays():
    """Create mock train and test arrays for model training."""
    np.random.seed(42)
    
    # 100 samples, 10 features
    X_train = np.random.randn(100, 10)
    y_train = X_train[:, 0] * 2 + X_train[:, 1] * 3 + np.random.randn(100) * 0.1
    
    X_test = np.random.randn(20, 10)
    y_test = X_test[:, 0] * 2 + X_test[:, 1] * 3 + np.random.randn(20) * 0.1
    
    train_arr = np.c_[X_train, y_train]
    test_arr = np.c_[X_test, y_test]
    
    return train_arr, test_arr


def test_model_trainer_runs():
    """Test that model trainer runs without crashing."""
    train_arr, test_arr = create_mock_train_test_arrays()
    
    trainer = ModelTrainer()
    score = trainer.initiate_model_trainer(train_arr, test_arr)
    
    assert isinstance(score, float)
    assert score > 0  # R² should be positive for this synthetic data


def test_model_trainer_returns_valid_r2():
    """Test that model trainer returns a valid R² score."""
    train_arr, test_arr = create_mock_train_test_arrays()
    
    trainer = ModelTrainer()
    score = trainer.initiate_model_trainer(train_arr, test_arr)
    
    # R² can be negative (overfitting) but should be a valid float
    assert isinstance(score, (float, np.floating))
    assert not np.isnan(score)
    assert not np.isinf(score)


def test_model_trainer_report_is_dict():
    """Test that evaluate_models returns a dictionary."""
    from src.utils import evaluate_models
    from sklearn.linear_model import Ridge
    
    train_arr, test_arr = create_mock_train_test_arrays()
    
    X_train = train_arr[:, :-1]
    y_train = train_arr[:, -1]
    X_test = test_arr[:, :-1]
    y_test = test_arr[:, -1]
    
    models = {"Ridge": Ridge()}
    params = {"Ridge": {"alpha": [0.1, 1.0]}}
    
    report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
    
    assert isinstance(report, dict)
    assert "Ridge" in report
    assert isinstance(report["Ridge"], float)


def test_config_cv_folds():
    """Test that CV folds configuration is valid."""
    assert CONFIG.model_trainer.cv_folds >= 2
    assert isinstance(CONFIG.model_trainer.cv_folds, int)


def test_config_min_score():
    """Test that minimum score threshold is valid."""
    assert 0 <= CONFIG.model_trainer.min_score_threshold <= 1
