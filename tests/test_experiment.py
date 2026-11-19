"""Tests for experiment module."""

import pandas as pd
from pathlib import Path
from src.Phase1 import get_model_pool


def test_get_model_pool():
    """Test model pool creation."""
    pool = get_model_pool()
    expected_models = [
        'LogisticRegression', 'DecisionTree', 'RandomForest', 'XGBoost', 'KNN',
        'NaiveBayes', 'AdaBoost', 'GradientBoosting', 'ExtraTrees', 'MLP',
    ]
    assert set(expected_models).issubset(set(pool.keys()))

    # Check that they are sklearn estimators
    for name, model in pool.items():
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')


def test_measure_energy_and_train():
    """Test energy measurement (mocked)."""
    # This would require mocking psutil and time, but for now skip full test
    pass


# Note: Full experiment testing would require actual datasets and is better done manually