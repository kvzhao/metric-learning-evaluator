"""Registered

  This table recording legal evaluation class.

"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))  # noqa


from metric_learning_evaluator.evaluator.evaluation_base import MetricEvaluationBase
from metric_learning_evaluator.evaluator.ranking_evaluation import RankingEvaluation
from metric_learning_evaluator.evaluator.facenet_evaluation import FacenetEvaluation

from metric_learning_evaluator.evaluator.standard_fields import EvaluationStandardFields as eval_fields

# NOTICE: Make sure each function passed correctness test. 
REGISTERED_EVALUATION_OBJECTS = {
    eval_fields.ranking: RankingEvaluation,
    eval_fields.facenet: FacenetEvaluation,
}

EVALUATION_DISPLAY_NAMES = {
    eval_fields.ranking: 'rank',
    eval_fields.facenet: 'pair',
}