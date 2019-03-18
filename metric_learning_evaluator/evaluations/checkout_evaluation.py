"""
  Customized Application for Checkout Evaluation


  NOTE @kv:
    This version is not mature and hard-coded.
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

import json
from pprint import pprint
from collections import defaultdict

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.data_tools.result_container import ResultContainer
from metric_learning_evaluator.data_tools.attribute_container  import AttributeContainer
from metric_learning_evaluator.data_tools.feature_object import FeatureDataObject

from metric_learning_evaluator.evaluations.evaluation_base import MetricEvaluationBase
from metric_learning_evaluator.evaluations.standard_fields import EvaluationStandardFields as eval_fields

from metric_learning_evaluator.metrics.standard_fields import MetricStandardFields as metric_fields
from metric_learning_evaluator.metrics.ranking_metrics import RankingMetrics

from metric_learning_evaluator.utils.distances import euclidean_distance
from metric_learning_evaluator.utils.distances import indexing_array
from metric_learning_evaluator.utils.sample_strategy import SampleStrategyStandardFields as sample_fields
from metric_learning_evaluator.utils.sample_strategy import SampleStrategy
from metric_learning_evaluator.query.standard_fields import AttributeStandardFields as attribute_fields

class CheckoutEvaluationStandardFields(object):
    # Some keys only used in ranking evaluation
    start = 'start'
    end = 'end'
    step = 'step'
    top_k_hit_accuracy = 'top_k_hit_accuracy'
    mAP = 'mAP'
    sampling = 'sampling'

    search_database = 'search_database'
    unseen_label_map= 'unseen_label_map'

checkout_fields = CheckoutEvaluationStandardFields

class CheckoutEvaluation(MetricEvaluationBase):

    def __init__(self, config):
        """Ranking Evaluation
        """
        super(CheckoutEvaluation, self).__init__(config)

        self._must_have_metrics = []

        # metrics with condition
        self._metric_with_threshold = [
            metric_fields.top_k_hit_accuracy,
        ]
        # metrics without condition
        self._metric_without_threshold = [
            metric_fields.mAP,
        ]

        print ('Create {}'.format(self._evaluation_name))
        self.show_configs()

    # metric_names
    @property
    def metric_names(self):
        # TODO @kv: make these easier
        _metric_names = []
        for _metric_name, _content in self._metrics.items():
            for _attr_name in ['seen', 'unseen']:
                if _content is None:
                    continue
                if _metric_name in self._metric_without_threshold:
                    _name = '{}-{}'.format(_attr_name, _metric_name)
                    _metric_names.append(_name)
                if _metric_name in self._metric_with_threshold:
                    # special condition
                    if _metric_name == metric_fields.top_k_hit_accuracy:
                        top_k_list = self._metrics[metric_fields.top_k_hit_accuracy]
                        for top_k in top_k_list:
                            _name = '{}-{}-@k={}'.format(_attr_name, _metric_name, top_k)
                            _metric_names.append(_name)
        return _metric_names

    def compute(self, embedding_container, attribute_container=None):
        result_container = ResultContainer()
        # Check whether attribute_container is given or not.
        if not attribute_container or attribute_fields.all_classes in self._attributes:

            ranking_config = self._metrics
            sample_config = self._configs[eval_fields.sampling]
            option_config = self._configs[eval_fields.option]

            # sampling configs
            class_sample_method = sample_config[sample_fields.class_sample_method]
            instance_sample_method = sample_config[sample_fields.instance_sample_method]
            num_of_db_instance = sample_config[sample_fields.num_of_db_instance]
            num_of_query_instance_per_class = sample_config[sample_fields.num_of_query_instance_per_class]
            num_of_query_class = sample_config[sample_fields.num_of_query_class]
            maximum_of_sampled_data = sample_config[sample_fields.maximum_of_sampled_data]

            # ids
            instance_ids = embedding_container.instance_ids
            label_ids = embedding_container.get_label_by_instance_ids(instance_ids)

            #option configs
            # TODO @kv: Load unseen label ids from given path.
            dataset_info_path = option_config[checkout_fields.unseen_label_map]
            with open(dataset_info_path, 'r') as fp:
                dataset_info = json.load(fp)
            unseen_dataset_info = dataset_info['unseen']
            unseen_label_ids = [unseen_data['label'] for unseen_data in unseen_dataset_info]
            unseen_instance_ids = embedding_container.get_instance_ids_by_label_ids(unseen_label_ids)

            seen_label_ids = [label for label in label_ids if not label in unseen_label_ids]
            seen_instance_ids = embedding_container.get_instance_ids_by_label_ids(seen_label_ids)
            """
              The Following Section should be a module.
            """
            for _attr in ['seen', 'unseen']:
                if _attr == 'seen':
                    sampler = SampleStrategy(seen_instance_ids, seen_label_ids)
                elif _attr == 'unseen':
                    sampler = SampleStrategy(unseen_instance_ids, unseen_label_ids)

                sampled = sampler.sample_query_and_database(
                    class_sample_method=class_sample_method,
                    instance_sample_method=instance_sample_method,
                    num_of_db_instance=num_of_db_instance,
                    num_of_query_class=num_of_query_class,
                    num_of_query_instance_per_class=num_of_query_instance_per_class,
                    maximum_of_sampled_data=maximum_of_sampled_data)

                query_embeddings = embedding_container.get_embedding_by_instance_ids(
                    sampled[sample_fields.query_instance_ids])
                query_label_ids = sampled[sample_fields.query_label_ids]

                db_embeddings = embedding_container.get_embedding_by_instance_ids(
                    sampled[sample_fields.db_instance_ids])
                db_label_ids = sampled[sample_fields.db_label_ids]

                # TODO @kv: type conversion at proper moment.
                query_label_ids = np.asarray(query_label_ids)
                db_label_ids = np.asarray(db_label_ids)

                # ranking configs
                top_k_list = ranking_config[metric_fields.top_k_hit_accuracy]

                # TODO @kv: Add more evaluation and reranking methods
                for top_k in top_k_list:

                    if top_k == 1:
                        continue

                    ranking_metrics = RankingMetrics(top_k)
                    hit_arrays = np.empty((query_embeddings.shape[0], top_k), dtype=np.bool)

                    for _idx, (_query_embed, _query_label) in enumerate(zip(query_embeddings, query_label_ids)):
                        # naive search
                        distances = euclidean_distance(_query_embed, db_embeddings)
                        sorted_distances = indexing_array(distances, distances)
                        indexed_db_labels = indexing_array(distances, db_label_ids)
                        print('Query label: {}'.format(_query_label))
                        print('Retrieved labels: {}'.format(indexed_db_labels[:10]))
                        print(sorted_distances[:20])
                        hits = indexed_db_labels[:top_k] == _query_label
                        hit_arrays[_idx, ...] = hits
                        if not hits[0]:
                            print('Retrieved labels: {} - GT label: {} Top1 Hit: {}'.format(indexed_db_labels, _query_label, hits[0]))

                    ranking_metrics.add_inputs(hit_arrays)
                    result_container.add(_attr, checkout_fields.top_k_hit_accuracy,
                                        ranking_metrics.topk_hit_accuracy, condition={'k': top_k})

                result_container.add(_attr, checkout_fields.top_k_hit_accuracy,
                                    ranking_metrics.topk_hit_accuracy, condition={'k': 1})
                result_container.add(_attr, checkout_fields.mAP,
                                    ranking_metrics.mean_average_precision)

            return result_container
        else:
            # with attribute filter
            raise NotImplementedError