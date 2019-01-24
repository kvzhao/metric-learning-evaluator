"""
    The Example of the EvaluationObject
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from evaluator.data_container import EmbeddingContainer
from evaluator.data_container import AttributeContainer
from evaluator.evaluation_base import MetricEvaluationBase


class MockStandardFields(object):
    # metric type
    distance = 'distance' # mock metric type

class MockEvaluation(MetricEvaluationBase):

    def __init__(self, per_eval_config):
        super(MockEvaluation, self).__init__(per_eval_config)

        """Mock Evaluation
            This mock evaluation functional object is mainly used in testing.

            * Check dimension when created
            * Execute 'mock_metric' functions when computation

        """
        print ('Create {}'.format(self._evaluation_name))
        # this will be called at builder


    def compute(self, embedding_container, attribute_container=None):
        _img_ids = embedding_container.image_ids
        _embeddings = embedding_container.embeddings
        _groups = attribute_container.group