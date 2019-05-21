"""Registered

  This table recording legal evaluation class.

"""
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))  # noqa

# ===========  Evaluations =============
from metric_learning_evaluator.evaluations.evaluation_base import MetricEvaluationBase
from metric_learning_evaluator.evaluations.ranking_evaluation import RankingEvaluation
from metric_learning_evaluator.evaluations.facenet_evaluation import FacenetEvaluation
from metric_learning_evaluator.evaluations.checkout_evaluation import CheckoutEvaluation
from metric_learning_evaluator.evaluations.ranking_with_attributes_evaluation import RankingWithAttributesEvaluation


# ===========  Index Agents =============
from metric_learning_evaluator.index.hnsw_agent import HNSWAgent
from metric_learning_evaluator.index.np_agent import NumpyAgent

# ===========  Query Interface =============
from metric_learning_evaluator.query.native_wrapper import NativeWrapper

## standard fields
from metric_learning_evaluator.config_parser.standard_fields import ConfigStandardFields as config_fields
from metric_learning_evaluator.evaluations.standard_fields import EvaluationStandardFields as eval_fields
from metric_learning_evaluator.query.standard_fields import QueryDatabaseStandardFields as query_fields


# NOTICE: Make sure each function passed correctness test. 
REGISTERED_EVALUATION_OBJECTS = {
    eval_fields.ranking: RankingEvaluation,
    eval_fields.facenet: FacenetEvaluation,
    eval_fields.checkout: CheckoutEvaluation,
    eval_fields.ranking_with_attrs: RankingWithAttributesEvaluation,
}
# NOTE: `CheckoutEvaluation` & `RankingWithAttributesEvaluation` will soon be deprecated.

EVALUATION_DISPLAY_NAMES = {
    eval_fields.ranking: 'rank',
    eval_fields.facenet: 'pair',
    eval_fields.checkout: 'checkout',
    eval_fields.ranking_with_attrs: 'rank_attrs',
}


REGISTERED_INDEX_AGENT = {
    config_fields.numpy_agent: NumpyAgent,
    config_fields.hnsw_agent: HNSWAgent,
}

REGISTERED_DATABASE_TYPE = {
    query_fields.native: NativeWrapper,
}