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

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.data_tools.result_container import ResultContainer

from metric_learning_evaluator.evaluations.evaluation_base import MetricEvaluationBase
from metric_learning_evaluator.metrics.ranking_metrics import RankingMetrics

from metric_learning_evaluator.index.agent import IndexAgent

from metric_learning_evaluator.utils.sample_strategy import SampleStrategy

from metric_learning_evaluator.core.standard_fields import MetricStandardFields as metric_fields
from metric_learning_evaluator.core.standard_fields import EvaluationStandardFields as eval_fields
from metric_learning_evaluator.core.standard_fields import AttributeStandardFields as attr_fields
from metric_learning_evaluator.core.standard_fields import SampleStrategyStandardFields as sample_fields
from metric_learning_evaluator.core.standard_fields import RankingEvaluationStandardFields as ranking_fields


class RankingEvaluation(MetricEvaluationBase):

    def __init__(self, config, mode=None):
        """Ranking Evaluation
            TODO with Attributes
          Two kinds of attribute
            - grouping
            - cross reference
        """
        super(RankingEvaluation, self).__init__(config, mode)

        self._must_have_metrics = []
        self._default_values = {
            metric_fields.distance_threshold:
            {
                ranking_fields.start: 0.5,
                ranking_fields.end: 1.5,
                ranking_fields.step: 0.2
            },
        }

        # metrics with condition
        self._metric_with_threshold = [
            metric_fields.top_k_hit_accuracy,
        ]
        # metrics without condition
        self._metric_without_threshold = [
            metric_fields.mAP,
        ]

        print('Create {}'.format(self.evaluation_name))
        self.show_configs()

    # metric_names
    @property
    def metric_names(self):
        # TODO @kv: make these easier
        _metric_names = []
        for _metric_name, _content in self.metrics.items():
            for _attr_name in self.attributes:
                if _content is None:
                    continue
                if _metric_name in self._metric_without_threshold:
                    _name = '{}/{}'.format(_attr_name, _metric_name)
                    _metric_names.append(_name)
                if _metric_name in self._metric_with_threshold:
                    # special condition
                    if _metric_name == metric_fields.top_k_hit_accuracy:
                        top_k_list = self.metrics[metric_fields.top_k_hit_accuracy]
                        for top_k in top_k_list:
                            _name = '{}/{}@k={}'.format(_attr_name, _metric_name, top_k)
                            _metric_names.append(_name)
        return _metric_names

    def compute(self, embedding_container):
        """Compute function
          Args:
            embedding_container: EmbeddingContainer
          Return:
            result_container: ResultContainer
        """

        # NOTE: Set result container as internal object
        self.result_container = ResultContainer()

        # ===== Groupings =====
        for group_cmd in self.group_commands:
            instance_ids_given_attribute = \
                embedding_container.get_instance_id_by_group_command(group_cmd)
            if len(instance_ids_given_attribute) == 0:
                continue
            label_ids_given_attribute = \
                embedding_container.get_label_by_instance_ids(instance_ids_given_attribute)

            # TODO: Can we do k-fold? only if the function pass result back
            self._sample_and_rank(group_cmd, instance_ids_given_attribute,
                                  label_ids_given_attribute, embedding_container)

        # ====== Cross References =====
        for cref_cmd in self.cross_reference_commands:
            query_instance_ids, database_instance_ids = \
                embedding_container.get_instance_id_by_cross_reference_command(cref_cmd)
            if len(query_instance_ids) == 0 or len(database_instance_ids) == 0:
                continue
            query_embeddings = embedding_container.get_embedding_by_instance_ids(query_instance_ids)
            query_label_ids = embedding_container.get_label_by_instance_ids(query_instance_ids)
            database_embeddings = embedding_container.get_embedding_by_instance_ids(database_instance_ids)
            database_label_ids = embedding_container.get_label_by_instance_ids(database_instance_ids)
            print('cross reference: {} with {} queries & {} databases'.format(
                cref_cmd, query_embeddings.shape[0], database_embeddings.shape[0]))

            self._rank(cref_cmd,
                       embedding_container,
                       query_instance_ids,
                       query_label_ids,
                       query_embeddings,
                       database_instance_ids,
                       database_label_ids,
                       database_embeddings,)

        return self.result_container

    def _rank(self,
              attr_name,
              embedding_container,
              query_instance_ids,
              query_label_ids,
              query_embeddings,
              database_instance_ids,
              database_label_ids,
              database_embeddings,):

        agent_type = self.configs.agent_type
        agent = IndexAgent(agent_type, database_instance_ids, database_embeddings)

        ranking_config = self.metrics
        # Search in batch
        top_k_list = ranking_config[metric_fields.top_k_hit_accuracy]
        max_k = max(top_k_list)

        # shape (N, max_K)
        retrieved_database_instance_ids, retrieved_database_distances = agent.search(query_embeddings, top_k=max_k)

        for top_k in top_k_list:
            if top_k == 1:
                continue
            ranking_metrics = RankingMetrics(top_k)
            hit_arrays = np.empty((query_embeddings.shape[0], top_k), dtype=np.bool)
            for _idx, (query_label_id, query_inst_id) in enumerate(zip(query_label_ids, query_instance_ids)):
                retrieved_instances = retrieved_database_instance_ids[_idx]
                retrieved_labels = embedding_container.get_label_by_instance_ids(retrieved_instances)
                hits = retrieved_labels[:top_k] == np.asarray(query_label_id)
                hit_arrays[_idx, ...] = hits

                # Hard-coded: print failure cases
                if self.mode == 'offline':
                    # record failure case
                    if not hits[0]:
                        retrieved_distances = retrieved_database_distances[_idx]
                        _event = {
                            'query_label': int(query_label_id),
                            'query_instance': int(query_inst_id),
                            'retrieved_labels': retrieved_labels[:top_k],
                            'retrieved_instances': retrieved_instances[:top_k].tolist(),
                            'retrieved_distances': retrieved_distances[:top_k].tolist()}
                        self.result_container.add_event(_event)
            ranking_metrics.add_inputs(hit_arrays)
            self.result_container.add(attr_name, ranking_fields.top_k_hit_accuracy,
                                      ranking_metrics.topk_hit_accuracy, condition={'k': top_k})
        # top 1 and mAP
        self.result_container.add(attr_name, ranking_fields.top_k_hit_accuracy,
                                  ranking_metrics.top1_hit_accuracy, condition={'k': 1})
        self.result_container.add(attr_name, ranking_fields.mAP,
                                  ranking_metrics.mean_average_precision)

    def _sample_and_rank(self,
                         attr_name,
                         instance_ids,
                         label_ids,
                         embedding_container,):
        """
          Args:
            instance_ids: List of integers
            label_ids: List of integers
            embedding_container:
                The EmbeddingContainer object
            attr_name:
                A string, as the command used for attribute container
            sample_config:
                Dict, obtain from self.sampling or self.config.sampling_section
            ranking_config:
                Dict, obtain from self.metrics, or self.configs.metric_section
            agent_type:
                A string, obtain from self.configs.agent_type
          Returns: None.
            This function will directly push the result in member result container object.
        """
        sampling_config = self.sampling

        sampler = SampleStrategy(instance_ids, label_ids)
        sampled = sampler.sample_query_and_database(
            class_sample_method=sampling_config[sample_fields.class_sample_method],
            instance_sample_method=sampling_config[sample_fields.instance_sample_method],
            num_of_db_instance_per_class=sampling_config[sample_fields.num_of_db_instance_per_class],
            num_of_query_class=sampling_config[sample_fields.num_of_query_class],
            num_of_query_instance_per_class=sampling_config[sample_fields.num_of_query_instance_per_class],
            maximum_of_sampled_data=sampling_config[sample_fields.maximum_of_sampled_data],)

        sampled_query_instance_ids = sampled[sample_fields.query_instance_ids]
        sampled_query_label_ids = sampled[sample_fields.query_label_ids]
        sampled_database_instance_ids = sampled[sample_fields.database_instance_ids]
        sampled_database_label_ids = sampled[sample_fields.database_label_ids]

        query_embeddings = embedding_container.get_embedding_by_instance_ids(sampled_query_instance_ids)
        database_embeddings = embedding_container.get_embedding_by_instance_ids(sampled_database_instance_ids)

        sampled_query_instance_ids = np.asarray(sampled_query_instance_ids)
        sampled_query_label_ids = np.asarray(sampled_query_label_ids)
        sampled_database_instance_ids = np.asarray(sampled_database_instance_ids)
        sampled_database_label_ids = np.asarray(sampled_database_label_ids)

        self._rank(attr_name,
                   embedding_container,
                   sampled_query_instance_ids,
                   sampled_query_label_ids,
                   query_embeddings,
                   sampled_database_instance_ids,
                   sampled_database_label_ids,
                   database_embeddings,)
