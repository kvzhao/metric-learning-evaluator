"""
    The Example of the EvaluationObject
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.data_tools.attribute_container import AttributeContainer
from metric_learning_evaluator.evaluations.evaluation_base import MetricEvaluationBase

from metric_learning_evaluator.metrics.standard_fields import MetricStandardFields as metric_fields


class MockStandardFields(object):
    # metric type
    distance = 'distance' # mock metric type

class MockEvaluation(MetricEvaluationBase):

    def __init__(self, config):
        super(MockEvaluation, self).__init__(config)

        """Mock Evaluation
            This mock evaluation functional object is mainly used in testing.

            * Check dimension when created
            * Execute 'mock_metric' functions when computation

        """
        print ('Create {}'.format(self._evaluation_name))
        # this will be called at builder

        self._available_metrics = [
            metric_fields.mAP,
            metric_fields.top_k,
        ]


    def compute(self, embedding_container, attribute_container=None):

        img_ids = embedding_container.image_ids
        embeddings = embedding_container.embeddings

        if attribute_container:
            groups = attribute_container.groups

        per_eval_attributes = self._config.get_per_eval_attributes(self.evaluation_name)
        per_eval_metrics = self._config.get_per_eval_metrics(self.evaluation_name)


        return {}