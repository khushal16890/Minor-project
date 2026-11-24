"""
Unit tests for supervised Machine Learning detection engine.
"""
import pytest
import numpy as np
import pandas as pd
from detection.supervised.trainer import ModelTrainer
from detection.supervised.predictor import ModelPredictor
from detection.supervised.evaluator import ModelEvaluator


def test_trainer_and_predictor_binary(sample_df):
    trainer = ModelTrainer()
    X = sample_df.drop(columns=["Label"])
    y = sample_df["Label"]

    train_res = trainer.train(
        X, y,
        model_name="test_xgb_binary",
        dataset_name="synthetic",
        apply_smote=False,  # small sample size in test
        save=True
    )
    assert train_res["model_name"] == "test_xgb_binary"
    assert "model_path" in train_res

    # Predict
    predictor = ModelPredictor()
    sample = X.iloc[0].to_dict()
    pred_res = predictor.predict("test_xgb_binary", sample)

    assert "predicted_label" in pred_res
    assert "confidence" in pred_res
    assert 0.0 <= pred_res["confidence"] <= 1.0
    assert "inference_latency_ms" in pred_res
    assert pred_res["inference_latency_ms"] >= 0.0


def test_evaluator_metrics(sample_df):
    evaluator = ModelEvaluator()
    X = sample_df.drop(columns=["Label"])
    y = sample_df["Label"]

    # Evaluate the saved test model
    eval_res = evaluator.evaluate(
        model_name="test_xgb_binary",
        X=X,
        y=y,
        dataset_name="synthetic_test"
    )
    assert "accuracy" in eval_res
    assert "f1_macro" in eval_res
    assert "precision_macro" in eval_res
    assert "recall_macro" in eval_res
    assert "confusion_matrix" in eval_res
    assert 0.0 <= eval_res["accuracy"] <= 1.0
