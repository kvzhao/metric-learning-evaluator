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

class RankingEvaluation(MetricEvaluationBase):

    def __init__(self, per_eval_config):
        super(RankingEvaluation, self).__init__(per_eval_config)

        print ('Create {}'.format(self._evaluation_name))
        # What we should do here?

    def compute(self, embedding_container, attribute_container=None):
        # _groups = self.attribute_container.group
        pass