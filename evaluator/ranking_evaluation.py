"""
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import math
import random
import itertools
import numpy as np
from random import shuffle

from evaluator.data_container import EmbeddingContainer
from evaluator.data_container import AttributeContainer
from evaluator.evaluation_base import MetricEvaluationBase

from metrics.ranking_metrics import RankingMetrics

from core.eval_standard_fields import MetricStandardFields as metric_fields
from core.eval_standard_fields import AttributeStandardFields as attribute_fields

from evaluator.sample_strategy import SampleStrategy
from evaluator.sample_strategy import SampleStrategyStandardFields as sample_fields

class RankingEvaluationStandardFields(object):
    # used for distance threshold
    start = 'start'
    end = 'end'
    step = 'step'
    top_k_hit_accuracy = 'top_k_hit_accuracy'
    sampling = 'sampling'

ranking_fields = RankingEvaluationStandardFields

class RankingEvaluation(MetricEvaluationBase):

    def __init__(self, config):
        """Ranking Evaluation
        """
        super(RankingEvaluation, self).__init__(config)

        self._must_have_metrics = []
        self._default_values = {
            metric_fields.distance_threshold: {
               ranking_fields.start: 0.5,
               ranking_fields.end: 1.5,
               ranking_fields.step: 0.2},
        }

        print ('Create {}'.format(self._evaluation_name))

    def compute(self, embedding_container, attribute_container=None):


        instance_ids = embedding_container.instance_ids
        label_ids = embedding_container.get_label_by_instance_ids(instance_ids)

        print (self._eval_metrics)
        ranking_config = self._eval_metrics[metric_fields.ranking]
        # Sampler
        class_sample_method = ranking_config[sample_fields.class_sample_method]
        instance_sample_method = ranking_config[sample_fields.instance_sample_method]

        ratio_of_sampled_class = ranking_config[sample_fields.ratio_of_sampled_class]
        #ratio_of_instance_per_class = ranking_config[sample_fields.ratio_of_instance_per_class]
        num_of_sampled_class = ranking_config[sample_fields.num_of_sampled_class]
        num_of_sampled_instance_per_class = ranking_config[sample_fields.num_of_sampled_instance_per_class]
        maximum_of_sampled_data = ranking_config[sample_fields.maximum_of_sampled_data]

        sampler = SampleStrategy(instance_ids, label_ids)

        sampler.sample_queries(class_sample_method,
                               instance_sample_method,
                               ratio_of_sampled_class,
                               num_of_sampled_class,
                               num_of_sampled_instance_per_class,
                               maximum_of_sampled_data)
