"""
Bias Detection & Mitigation

Detect, measure, and mitigate data biases that could propagate into the recommendation engine.

DataSlicer: slice data by demographic/categorical features
BiasDetector: fairness metrics via Fairlearn + custom metrics
BiasMitigator: resampling, SMOTE, threshold adjustment
"""

from src.data_pipeline.bias_detection.slicer import DataSlicer
from src.data_pipeline.bias_detection.metrics import BiasDetector, BiasReport
from src.data_pipeline.bias_detection.mitigation import BiasMitigator

__all__ = [
    "DataSlicer",
    "BiasDetector",
    "BiasReport",
    "BiasMitigator",
]
