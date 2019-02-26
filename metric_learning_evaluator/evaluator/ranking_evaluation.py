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


from metric_learning_evaluator.evaluator.data_container import EmbeddingContainer
from metric_learning_evaluator.evaluator.data_container import AttributeContainer
from metric_learning_evaluator.evaluator.data_container import ResultContainer
from metric_learning_evaluator.evaluator.evaluation_base import MetricEvaluationBase

from metric_learning_evaluator.metrics.ranking_metrics import RankingMetrics
from metric_learning_evaluator.metrics.distances import euclidean_distance
from metric_learning_evaluator.metrics.distances import indexing_array

from metric_learning_evaluator.evaluator.sample_strategy import SampleStrategy

from metric_learning_evaluator.core.eval_standard_fields import MetricStandardFields as metric_fields
from metric_learning_evaluator.core.eval_standard_fields import AttributeStandardFields as attribute_fields
from metric_learning_evaluator.evaluator.sample_strategy import SampleStrategyStandardFields as sample_fields

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

        result_container = ResultContainer(self._eval_metrics, self._eval_attributes)
        # Check whether attribute_container is given or not.
        if not attribute_container or attribute_fields.all_classes in self._eval_attributes:
            instance_ids = embedding_container.instance_ids
            label_ids = embedding_container.get_label_by_instance_ids(instance_ids)
            ranking_config = self._eval_metrics[metric_fields.ranking]

            # sampling configs
            class_sample_method = ranking_config[sample_fields.class_sample_method]
            instance_sample_method = ranking_config[sample_fields.instance_sample_method]
            num_of_db_instance = ranking_config[sample_fields.num_of_db_instance]
            num_of_query_instance = ranking_config[sample_fields.num_of_query_instance]
            num_of_query_class = ranking_config[sample_fields.num_of_query_class]
            maximum_of_sampled_data = ranking_config[sample_fields.maximum_of_sampled_data]
            # ranking configs
            top_k = ranking_config[ranking_fields.top_k_hit_accuracy]

            sampler = SampleStrategy(instance_ids, label_ids)
            sampled = sampler.sample_query_and_database(
                class_sample_method=class_sample_method,
                instance_sample_method=instance_sample_method,
                num_of_db_instance=num_of_db_instance,
                num_of_query_class=num_of_query_class,
                num_of_query_instance=num_of_query_instance,
                maximum_of_sampled_data=maximum_of_sampled_data
            )

            query_embeddings = embedding_container.get_embedding_by_instance_ids(
                sampled[sample_fields.query_instance_ids])
            query_label_ids = sampled[sample_fields.query_label_ids]

            db_embeddings = embedding_container.get_embedding_by_instance_ids(
                sampled[sample_fields.db_instance_ids])
            db_label_ids = sampled[sample_fields.db_label_ids]

            print(query_embeddings.shape, db_embeddings.shape)
            print(len(query_label_ids), len(db_label_ids))

            # TODO @kv: type conversion at proper moment.
            query_label_ids = np.asarray(query_label_ids)
            db_label_ids = np.asarray(db_label_ids)

            ranking_metrics = RankingMetrics(top_k)
            hit_arrays = np.empty((query_embeddings.shape[0], top_k), dtype=np.bool)

            for _idx, (_query_embed, _query_label) in enumerate(zip(query_embeddings, query_label_ids)):

                distances = euclidean_distance(_query_embed, db_embeddings)

                indexed_query_label = indexing_array(distances, db_label_ids)

                hits = indexed_query_label[:top_k] == _query_label

                hit_arrays[_idx, ...] = hits

            ranking_metrics.add_inputs(hit_arrays)
            result_container.add(attribute_fields.all_classes, ranking_fields.top_k_hit_accuracy,
                                 top_k, ranking_metrics.topk_hit_accuracy)
            result_container.add(attribute_fields.all_classes, 'top_1_hit_accuracy',
                                 1, ranking_metrics.top1_hit_accuracy)

            return result_container
        else:
            # with attribute filter
            pass
