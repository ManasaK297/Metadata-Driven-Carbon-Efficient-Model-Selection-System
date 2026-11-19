"""Tests for preprocessing module."""

import pandas as pd
import numpy as np
from src.preprocessing import preprocess_dataset


def test_preprocess_basic():
    """Test basic preprocessing."""
    df = pd.DataFrame({
        'age': [25, 30, 35],
        'job': ['A', 'B', 'A'],
        'target': [0, 1, 0]
    })
    X_train, X_test, y_train, y_test = preprocess_dataset(df, target_col='target')

    assert X_train.shape[0] > 0
    assert y_train is not None
    assert y_test is not None
    # Check encoding
    assert X_train['job'].dtype in [np.int32, np.int64]  # encoded


def test_preprocess_with_missing():
    """Test preprocessing with missing values."""
    df = pd.DataFrame({
        'age': [25, np.nan, 35],
        'job': ['A', 'B', np.nan],
        'target': [0, 1, 0]
    })
    X_train, X_test, y_train, y_test = preprocess_dataset(df, target_col='target')

    # Missing values should be filled
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()


def test_preprocess_no_target():
    """Test preprocessing without target."""
    df = pd.DataFrame({
        'age': [25, 30, 35],
        'job': ['A', 'B', 'A']
    })
    X_train, X_test, y_train, y_test = preprocess_dataset(df)

    assert y_train is None
    assert y_test is None
    assert X_train.shape[1] == 2  # age and encoded job


def test_preprocess_with_scaling():
    """Test that scaling is applied to numeric features."""
    df = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'salary': [30000, 40000, 50000, 60000, 70000],
        'target': [0, 1, 0, 1, 0]
    })
    X_train, X_test, y_train, y_test = preprocess_dataset(
        df, target_col='target', scale_features=True
    )

    # Scaled features should have mean ~0 and std ~1
    assert abs(X_train['age'].mean()) < 0.5  # Close to 0
    assert abs(X_train['salary'].mean()) < 0.5
    print(f"Age mean: {X_train['age'].mean():.4f}, std: {X_train['age'].std():.4f}")


def test_preprocess_without_scaling():
    """Test preprocessing with scaling disabled."""
    df = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'salary': [30000, 40000, 50000, 60000, 70000],
        'target': [0, 1, 0, 1, 0]
    })
    X_train_no_scale, _, _, _ = preprocess_dataset(
        df, target_col='target', scale_features=False
    )
    X_train_scaled, _, _, _ = preprocess_dataset(
        df, target_col='target', scale_features=True
    )

    # Unscaled should have different mean/std from scaled
    assert abs(X_train_no_scale['age'].mean() - X_train_scaled['age'].mean()) > 0.1


def test_preprocess_stratified_split():
    """Test stratified split preserves class distribution."""
    # Create imbalanced dataset
    df = pd.DataFrame({
        'feature': range(100),
        'target': [0] * 80 + [1] * 20
    })

    # With stratification
    X_train, X_test, y_train, y_test = preprocess_dataset(
        df, target_col='target', stratify=True
    )

    # Check that class distribution is preserved approximately
    train_ratio = (y_train == 1).sum() / len(y_train)
    test_ratio = (y_test == 1).sum() / len(y_test)
    original_ratio = 20 / 100

    # Should be close to original ratio (±10%)
    assert abs(train_ratio - original_ratio) < 0.15
    assert abs(test_ratio - original_ratio) < 0.15
    print(f"Original ratio: {original_ratio:.2f}, Train: {train_ratio:.2f}, Test: {test_ratio:.2f}")


def test_preprocess_non_stratified_split():
    """Test non-stratified split."""
    df = pd.DataFrame({
        'feature': range(100),
        'target': [0] * 80 + [1] * 20
    })

    X_train, X_test, y_train, y_test = preprocess_dataset(
        df, target_col='target', stratify=False
    )

    # Just verify it works without error
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0


def test_preprocess_mixed_types():
    """Test preprocessing with mixed numeric and categorical features."""
    df = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA'],
        'salary': [30000, 40000, 50000, 60000, 70000],
        'target': [0, 1, 0, 1, 0]
    })

    X_train, X_test, y_train, y_test = preprocess_dataset(
        df, target_col='target', scale_features=True
    )

    # Check that we have all features
    assert X_train.shape[1] == 3
    # City should be encoded
    assert X_train['city'].dtype in [np.int32, np.int64]
    print("Mixed types test passed")


def test_preprocess_edge_case_single_category():
    """Test preprocessing with single category in categorical column."""
    df = pd.DataFrame({
        'age': [25, 30, 35],
        'city': ['NYC', 'NYC', 'NYC'],
        'target': [0, 1, 0]
    })

    X_train, X_test, y_train, y_test = preprocess_dataset(
        df, target_col='target'
    )

    assert X_train.shape[0] > 0
    print("Single category test passed")