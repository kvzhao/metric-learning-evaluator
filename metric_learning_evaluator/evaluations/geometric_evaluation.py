import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from metric_learning_evaluator.evaluations.evaluation_base import MetricEvaluationBase

class GeometricEvaluation(MetricEvaluationBase):

    def __init__(self):
        pass