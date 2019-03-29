"""
"""

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import json
import math
import random
import itertools
import numpy as np
from random import shuffle

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.data_tools.result_container import ResultContainer
from metric_learning_evaluator.data_tools.attribute_container import AttributeContainer

from metric_learning_evaluator.evaluations.evaluation_base import MetricEvaluationBase
from metric_learning_evaluator.evaluations.standard_fields import EvaluationStandardFields as eval_fields

from metric_learning_evaluator.metrics.standard_fields import MetricStandardFields as metric_fields
from metric_learning_evaluator.metrics.ranking_metrics import RankingMetrics

from metric_learning_evaluator.utils.distances import euclidean_distance
from metric_learning_evaluator.utils.distances import indexing_array
from metric_learning_evaluator.utils.sample_strategy import SampleStrategyStandardFields as sample_fields
from metric_learning_evaluator.utils.sample_strategy import SampleStrategy
from metric_learning_evaluator.utils.attributes_groups_divider import AttributesGroupsDivider
from metric_learning_evaluator.utils.attributes_groups_divider import AttributesGroupsDividerFields as divider_fields
from metric_learning_evaluator.query.standard_fields import AttributeStandardFields as attribute_fields


class RankingEvaluationStandardFields(object):
    # Some keys only used in ranking evaluation
    start = 'start'
    end = 'end'
    step = 'step'
    top_k_hit_accuracy = 'top_k_hit_accuracy'
    mAP = 'mAP'
    sampling = 'sampling'


ranking_fields = RankingEvaluationStandardFields


class RankingWithAttributesEvaluation(MetricEvaluationBase):

    def __init__(self, config):
        """Ranking Evaluation
        """
        super(RankingWithAttributesEvaluation, self).__init__(config)

        self._must_have_metrics = []
        self._default_values = {
            metric_fields.distance_threshold: {
                ranking_fields.start: 0.5,
                ranking_fields.end: 1.5,
                ranking_fields.step: 0.2},
        }

        # metrics with condition
        self._metric_with_threshold = [
            metric_fields.top_k_hit_accuracy,
        ]
        # metrics without condition
        self._metric_without_threshold = [
            metric_fields.mAP,
        ]

        print('Create {}'.format(self._evaluation_name))

        grouping_rule_file_path = self._configs[eval_fields.attribute].get(divider_fields.grouping_rules, '')
        if not os.path.exists(grouping_rule_file_path):
            raise FileNotFoundError("grouping_rule {} not found".format(grouping_rule_file_path))
        with open(grouping_rule_file_path) as f:
            self._grouping_rules = json.load(f)
        print("grouping_rule {} is loaded".format(grouping_rule_file_path))

        self.show_configs()

    # metric_names
    @property
    def metric_names(self):
        # TODO @kv: make these easier
        _metric_names = []
        for _metric_name, _content in self._metrics.items():
            for _group in self._grouping_rules.get(divider_fields.rank_events):
                comment = _group[divider_fields.comment]
                if _content is None:
                    continue
                if _metric_name in self._metric_without_threshold:
                    _name = '{}/{}'.format(comment, _metric_name)
                    _metric_names.append(_name)
                if _metric_name in self._metric_with_threshold:
                    # special condition
                    if _metric_name == metric_fields.top_k_hit_accuracy:
                        top_k_list = self._metrics[metric_fields.top_k_hit_accuracy]
                        for top_k in top_k_list:
                            _name = '{}/{}@k={}'.format(comment, _metric_name, top_k)
                            _metric_names.append(_name)
        return _metric_names

    def compute(self, embedding_container, attribute_container=None):
        divider = AttributesGroupsDivider(attribute_container, self._grouping_rules)

        eval_groups = divider.groups
        eval_events = divider.rank_events

        sample_config = self._configs[eval_fields.sampling]

        # sampling configs
        class_sample_method = sample_config[sample_fields.class_sample_method]
        instance_sample_method = sample_config[sample_fields.instance_sample_method]
        num_of_db_instance = sample_config[sample_fields.num_of_db_instance]
        num_of_query_instance_per_class = sample_config[sample_fields.num_of_query_instance_per_class]
        num_of_query_class = sample_config[sample_fields.num_of_query_class]
        maximum_of_sampled_data = sample_config[sample_fields.maximum_of_sampled_data]

        # ranking configs
        ranking_config = self._metrics
        top_k_list = ranking_config[metric_fields.top_k_hit_accuracy]

        result_container = ResultContainer()
        for event in eval_events:
            event_comment = event[divider_fields.comment]
            query_group_name = event[divider_fields.query_group]
            db_group_name = event[divider_fields.db_group]

            query_instance_ids = []
            query_label_ids = []
            db_instance_ids = []
            db_label_ids = []

            if query_group_name == db_group_name:
                # if query and db are same group
                group_name = query_group_name
                instance_ids = eval_groups[group_name]
                label_ids = embedding_container.get_label_by_instance_ids(instance_ids)
                if len(instance_ids) > 1:
                    # Online sample mode:
                    sampler = SampleStrategy(instance_ids, label_ids)
                    sampled = sampler.sample_query_and_database(
                        class_sample_method=class_sample_method,
                        instance_sample_method=instance_sample_method,
                        num_of_db_instance=num_of_db_instance,
                        num_of_query_class=num_of_query_class,
                        num_of_query_instance_per_class=num_of_query_instance_per_class,
                        maximum_of_sampled_data=maximum_of_sampled_data
                    )

                    query_instance_ids = sampled[sample_fields.query_instance_ids]
                    query_label_ids = sampled[sample_fields.query_label_ids]
                    db_instance_ids = sampled[sample_fields.db_instance_ids]
                    db_label_ids = sampled[sample_fields.db_label_ids]

                else:
                    print("number of group instance is zero , skipping...")

            else:
                # if query and db are not same group
                query_instance_ids = eval_groups[query_group_name]
                db_instance_ids = eval_groups[db_group_name]
                inner_instance_ids = divider.inner_join(query_instance_ids, db_instance_ids)

                if len(inner_instance_ids) > 0:
                    print("Warning query group : {} and db group :{} is overlap with {} instances" \
                          ", block same instance in db".format(query_group_name, db_group_name,
                                                               len(inner_instance_ids)))

                query_instance_ids = query_instance_ids
                query_label_ids = embedding_container.get_label_by_instance_ids(query_instance_ids)
                db_instance_ids = list(set(db_instance_ids) - set(inner_instance_ids))
                db_label_ids = embedding_container.get_label_by_instance_ids(db_instance_ids)

            query_embeddings = embedding_container.get_embedding_by_instance_ids(query_instance_ids)
            db_embeddings = embedding_container.get_embedding_by_instance_ids(db_instance_ids)

            if len(set(query_label_ids) - set(db_label_ids)) == 0 and len(query_label_ids) > 0 and len(
                    db_label_ids) > 0:
                query_label_ids = np.asarray(query_label_ids)
                db_label_ids = np.asarray(db_label_ids)

                # print("embedding query : {},db:{}".format(len(query_embeddings), len(db_embeddings)))
                print("query : {},db:{}".format(len(query_label_ids), len(db_label_ids)))
                # event_comment = event_comment + "(query-{} db-{})".format(len(query_label_ids), len(db_label_ids))
                for top_k in top_k_list:
                    ranking_metrics = RankingMetrics(top_k)
                    hit_arrays = np.empty((query_embeddings.shape[0], top_k), dtype=np.bool)
                    for _idx, (_query_embed, _query_label) in enumerate(zip(query_embeddings, query_label_ids)):
                        distances = euclidean_distance(_query_embed, db_embeddings)

                        indexed_query_label = indexing_array(distances, db_label_ids)

                        hits = indexed_query_label[:top_k] == _query_label

                        hit_arrays[_idx, ...] = hits
                    ranking_metrics.add_inputs(hit_arrays)

                    result_container.add(event_comment, ranking_fields.top_k_hit_accuracy,
                                         ranking_metrics.topk_hit_accuracy, condition={'k': top_k})
                result_container.add(event_comment, ranking_fields.top_k_hit_accuracy,
                                     ranking_metrics.top1_hit_accuracy, condition={'k': 1})

                result_container.add(event_comment, ranking_fields.mAP,
                                     ranking_metrics.mean_average_precision)
            else:
                if len(query_label_ids) <= 0 or len(db_label_ids) <= 0:
                    print("Warning len(query_label_ids)<=0 or len(db_label_ids)<=0 , skipping...")
                else:
                    print("Warning classes of db is not included all query classes , skipping...")
                for top_k in top_k_list:
                    result_container.add(event_comment, ranking_fields.top_k_hit_accuracy,
                                         -1, condition={'k': top_k})
                result_container.add(event_comment, ranking_fields.mAP, -1)

        return result_container
