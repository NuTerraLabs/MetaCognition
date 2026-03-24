"""
Package initialization for Synthetic Metacognition

Exposes main classes and functions for easy import.
"""

from .models import (
    BaseLearner,
    MetaMonitor,
    MetaController,
    MetaCognitiveModel,
    BaselineMLP
)

from .training import (
    MetacognitiveLoss,
    MetacognitiveTrainer,
    train_baseline_model
)

from .evaluation import (
    MetacognitiveEvaluator,
    expected_calibration_error,
    brier_score,
    uncertainty_error_correlation,
    selective_accuracy,
    compare_models,
    print_comparison_table
)

from .reflection import (
    MetacognitiveAgent,
    Proposition,
    MetaProposition,
    ReflectionRules,
    Modality
)

__version__ = "1.0.0"
__author__ = "Anonymous"
__all__ = [
    # Models
    "BaseLearner",
    "MetaMonitor",
    "MetaController",
    "MetaCognitiveModel",
    "BaselineMLP",
    # Training
    "MetacognitiveLoss",
    "MetacognitiveTrainer",
    "train_baseline_model",
    # Evaluation
    "MetacognitiveEvaluator",
    "expected_calibration_error",
    "brier_score",
    "uncertainty_error_correlation",
    "selective_accuracy",
    "compare_models",
    "print_comparison_table",
    # Reflection
    "MetacognitiveAgent",
    "Proposition",
    "MetaProposition",
    "ReflectionRules",
    "Modality",
]
