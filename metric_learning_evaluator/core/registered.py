"""Registered

  This table recording legal evaluation class.

"""
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))  # noqa

from metric_learning_evaluator.evaluations.evaluation_base import MetricEvaluationBase
from metric_learning_evaluator.evaluations.ranking_evaluation import RankingEvaluation
from metric_learning_evaluator.evaluations.facenet_evaluation import FacenetEvaluation
from metric_learning_evaluator.evaluations.ranking_with_attributes_evaluation import RankingWithAttributesEvaluation

from metric_learning_evaluator.evaluations.standard_fields import EvaluationStandardFields as eval_fields

# NOTICE: Make sure each function passed correctness test. 
REGISTERED_EVALUATION_OBJECTS = {
    eval_fields.ranking: RankingEvaluation,
    eval_fields.facenet: FacenetEvaluation,
    eval_fields.ranking_with_attrs: RankingWithAttributesEvaluation,

}

EVALUATION_DISPLAY_NAMES = {
    eval_fields.ranking: 'rank',
    eval_fields.facenet: 'pair',
    eval_fields.ranking_with_attrs: 'rank_attrs',
}
