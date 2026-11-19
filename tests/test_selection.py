"""Tests for model_selector module."""

import pandas as pd
import numpy as np
from pathlib import Path
from src.model_selector import ModelSelector


def test_model_selector_init():
    """Test ModelSelector initialization with knowledge base."""
    selector = ModelSelector("knowledge_base.csv")
    assert selector.knowledge_base is not None
    assert len(selector.knowledge_base) > 0
    print(f"Knowledge base has {len(selector.knowledge_base)} entries")


def test_set_constraints():
    """Test setting constraints."""
    selector = ModelSelector("knowledge_base.csv")
    
    selector.set_constraints(min_accuracy=0.8, max_energy_kwh=0.1, max_co2_kg=0.000001)
    
    assert selector.constraints['min_accuracy'] == 0.8
    assert selector.constraints['max_energy_kwh'] == 0.1
    assert selector.constraints['max_co2_kg'] == 0.000001


def test_train_meta_model():
    """Test meta-model training."""
    selector = ModelSelector("knowledge_base.csv")
    selector.train_meta_model()
    
    assert selector.meta_model is not None
    assert selector.model_encoder is not None
    print("Meta-model trained successfully")


def test_predict_model_from_metadata():
    """Test model prediction from metadata."""
    selector = ModelSelector("knowledge_base.csv")
    selector.train_meta_model()
    
    # Sample metadata
    metadata = {
        "n_rows": 10000,
        "n_columns": 20,
        "n_numeric": 18,
        "n_categorical": 2,
        "missing_fraction": 0.05,
        "mean_cardinality": 500.0,
    }
    
    predicted_model = selector.predict_model_from_metadata(metadata)
    assert predicted_model in selector.knowledge_base['model_name'].unique()
    print(f"Predicted model: {predicted_model}")


def test_select_best_model_no_constraints():
    """Test model selection without constraints."""
    selector = ModelSelector("knowledge_base.csv")
    selector.train_meta_model()
    
    metadata = {
        "n_rows": 5000,
        "n_columns": 15,
        "n_numeric": 12,
        "n_categorical": 3,
        "missing_fraction": 0.02,
        "mean_cardinality": 100.0,
    }
    
    best_model, metrics, explanation = selector.select_best_model(metadata)
    
    assert best_model in selector.knowledge_base['model_name'].unique()
    assert 'accuracy' in metrics
    assert 'energy_kwh' in metrics
    assert 'co2_kg' in metrics
    assert 'Selected Model:' in explanation
    print(f"Selected model: {best_model}, Accuracy: {metrics['accuracy']:.4f}")


def test_select_best_model_with_constraints():
    """Test model selection with accuracy constraint."""
    selector = ModelSelector("knowledge_base.csv")
    selector.train_meta_model()
    
    # Set high accuracy requirement
    selector.set_constraints(min_accuracy=0.9)
    
    metadata = {
        "n_rows": 100000,
        "n_columns": 30,
        "n_numeric": 28,
        "n_categorical": 2,
        "missing_fraction": 0.01,
        "mean_cardinality": 1000.0,
    }
    
    best_model, metrics, explanation = selector.select_best_model(metadata)
    
    # Should respect constraint
    assert metrics['accuracy'] >= 0.9 or len(selector.knowledge_base) == 0
    assert 'Constraint Satisfaction:' in explanation
    print(f"Selected model with constraint: {best_model}, Accuracy: {metrics['accuracy']:.4f}")


def test_save_and_load_meta_model():
    """Test saving and loading meta-model."""
    import tempfile
    from pathlib import Path
    
    selector1 = ModelSelector("knowledge_base.csv")
    selector1.train_meta_model()
    
    # Save to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        selector1.save_meta_model(model_path)
        
        # Create new selector and load
        selector2 = ModelSelector("knowledge_base.csv")
        selector2.load_meta_model(model_path)
        
        assert selector2.meta_model is not None
        assert selector2.model_encoder is not None
        print("Meta-model saved and loaded successfully")


def test_explanation_generation():
    """Test that explanations are comprehensive."""
    selector = ModelSelector("knowledge_base.csv")
    selector.train_meta_model()
    selector.set_constraints(min_accuracy=0.8, max_energy_kwh=0.00001)
    
    metadata = {
        "n_rows": 10000,
        "n_columns": 20,
        "n_numeric": 18,
        "n_categorical": 2,
        "missing_fraction": 0.05,
        "mean_cardinality": 500.0,
    }
    
    best_model, metrics, explanation = selector.select_best_model(metadata)
    
    # Check explanation contains key sections
    assert "Selected Model:" in explanation
    assert "Dataset Characteristics:" in explanation
    assert "Meta-Model Prediction:" in explanation
    assert "Performance Metrics:" in explanation
    assert "Constraint Satisfaction:" in explanation
    assert "Accuracy:" in explanation
    assert "Energy:" in explanation
    assert "CO₂" in explanation
    print("Explanation contains all required sections")