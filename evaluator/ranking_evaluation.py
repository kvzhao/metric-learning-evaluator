"""
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from evaluator.data_container import EmbeddingContainer
from evaluator.data_container import AttributeContainer
from evaluator.evaluation_base import MetricEvaluationBase

class RankingEvaluation(MetricEvaluationBase):

    def __init__(self, config):
        """Ranking

        """
        super(RankingEvaluation, self).__init__(config)

        print ('Create {}'.format(self._evaluation_name))

    def compute(self, embedding_container, attribute_container=None):
        pass
