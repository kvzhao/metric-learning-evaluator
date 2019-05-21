"""
  Customized Application for Checkout Evaluation

  Evaluation Metrics:
    - Seen to Seen
    - Unseen to Unseen
    - Seen to Total
    - Unseen to Total

  Evaluation Report:
    - S-S, S-T, U-U, U-T
    - misclassified

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
from tqdm import tqdm

import json
from pprint import pprint
import pytablewriter
from collections import defaultdict

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.data_tools.result_container import ResultContainer
from metric_learning_evaluator.data_tools.attribute_container  import AttributeContainer
from metric_learning_evaluator.data_tools.feature_object import FeatureObject

from metric_learning_evaluator.evaluations.evaluation_base import MetricEvaluationBase
from metric_learning_evaluator.evaluations.standard_fields import EvaluationStandardFields as eval_fields

from metric_learning_evaluator.metrics.standard_fields import MetricStandardFields as metric_fields
from metric_learning_evaluator.metrics.ranking_metrics import RankingMetrics

from metric_learning_evaluator.index.utils import euclidean_distance
from metric_learning_evaluator.index.utils import indexing_array
from metric_learning_evaluator.utils.sample_strategy import SampleStrategyStandardFields as sample_fields
from metric_learning_evaluator.utils.sample_strategy import SampleStrategy
from metric_learning_evaluator.query.standard_fields import AttributeStandardFields as attr_fields

class CheckoutEvaluationStandardFields(object):
    # Some keys only used in ranking evaluation
    start = 'start'
    end = 'end'
    step = 'step'
    top_k_hit_accuracy = 'top_k_hit_accuracy'
    mAP = 'mAP'
    sampling = 'sampling'

    seen_unique_ids = 'seen_unique_ids'
    unseen_unique_ids = 'unseen_unique_ids'
    label_map = 'label_map'

    num_of_seen_query_class = 'num_of_seen_query_class'
    num_of_seen_db_instance = 'num_of_seen_db_instance'
    num_of_seen_query_instance_per_class = 'num_of_seen_query_instance_per_class'
    num_of_unseen_query_class = 'num_of_unseen_query_class'
    num_of_unseen_db_instance = 'num_of_unseen_db_instance'
    num_of_unseen_query_instance_per_class = 'num_of_unseen_query_instance_per_class'

    save_report = 'save_report'

checkout_fields = CheckoutEvaluationStandardFields


def load_json(json_path):
    try:
        with open(json_path, 'r') as fp:
            _dict = json.load(fp)
    except:
        raise IOError('Labelmap: {} can not be loaded.')

    return _dict

class Reporter(object):
    def __init__(self, table, label_map=None):
        self.table = table
        self.label_map = label_map

    def _parse(self):
        pass

    def save_report(self, save_path):

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(save_path + '/table_raw.json', 'w') as fp:
            json.dump(self.table, fp)

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
            for _attr_name in self._attributes:
                if _content is None:
                    continue
                if _metric_name in self._metric_without_threshold:
                    _name = '{}/{}'.format(_attr_name, _metric_name)
                    _metric_names.append(_name)
                if _metric_name in self._metric_with_threshold:
                    # special condition
                    if _metric_name == metric_fields.top_k_hit_accuracy:
                        top_k_list = self._metrics[metric_fields.top_k_hit_accuracy]
                        for top_k in top_k_list:
                            _name = '{}/{}@k={}'.format(_attr_name, _metric_name, top_k)
                            _metric_names.append(_name)
        return _metric_names

    def compute(self, embedding_container, attribute_container=None):
        result_container = ResultContainer()
        # Check whether attribute_container is given or not.

        ranking_config = self._metrics
        sample_config = self._configs[eval_fields.sampling]
        option_config = self._configs[eval_fields.option]

        # sampling configs
        class_sample_method = sample_config[sample_fields.class_sample_method]
        instance_sample_method = sample_config[sample_fields.instance_sample_method]

        # option configs used in checkout evaluation
        num_of_seen_query_class = option_config[checkout_fields.num_of_seen_query_class]
        num_of_seen_db_instance = option_config[checkout_fields.num_of_seen_db_instance]
        num_of_seen_query_instance_per_class = option_config[checkout_fields.num_of_seen_query_instance_per_class]
        num_of_unseen_query_class = option_config[checkout_fields.num_of_unseen_query_class]
        num_of_unseen_db_instance = option_config[checkout_fields.num_of_unseen_db_instance]
        num_of_unseen_query_instance_per_class = option_config[checkout_fields.num_of_unseen_query_instance_per_class]
        save_report = option_config[checkout_fields.save_report]

        if save_report is not None:
            report_table = {}

        # instance and label ids
        all_instance_ids = embedding_container.instance_ids
        all_label_ids = embedding_container.get_label_by_instance_ids(all_instance_ids)

        # path to label maps
        seen_id_path = option_config[checkout_fields.seen_unique_ids]
        unseen_id_path = option_config[checkout_fields.unseen_unique_ids]
        labelmap_path = option_config[checkout_fields.label_map]

        # load label maps
        seen_unique_id_map = load_json(seen_id_path)
        unseen_unique_id_map = load_json(unseen_id_path)
        # Used for per instance results
        standard_label_map = load_json(labelmap_path)

        seen_unique_ids = [int(k) for k in seen_unique_id_map.keys()]
        unseen_unique_ids = [int(k) for k in unseen_unique_id_map.keys()]

        seen_instance_ids, seen_label_ids = [], []
        unseen_instance_ids, unseen_label_ids = [], []
        for label_id, inst_id in zip(all_label_ids, all_instance_ids):
            if label_id in seen_unique_ids:
                seen_label_ids.append(label_id)
                seen_instance_ids.append(inst_id)
            elif label_id in unseen_unique_ids:
                unseen_label_ids.append(label_id)
                unseen_instance_ids.append(inst_id)
        print('Container has {} classes with {} instances, {} are seen ({}) and {} are unseen ({}).'.format(
            len(set(all_label_ids)), len(all_instance_ids), len(seen_instance_ids), len(set(seen_label_ids)), 
            len(unseen_instance_ids), len(set(unseen_label_ids))))

        print('Create Seen Data Sampler')
        seen_sampler = SampleStrategy(seen_instance_ids, seen_label_ids)
        sampled_seen = seen_sampler.sample_query_and_database(
                            class_sample_method=class_sample_method,
                            instance_sample_method=instance_sample_method,
                            num_of_db_instance=num_of_seen_db_instance,
                            num_of_query_class=num_of_seen_query_class,
                            num_of_query_instance_per_class=num_of_seen_query_instance_per_class,
                            maximum_of_sampled_data=1000000)

        print('Create Unseen Data Sampler')
        unseen_sampler = SampleStrategy(unseen_instance_ids, unseen_label_ids)
        sampled_unseen = unseen_sampler.sample_query_and_database(
                            class_sample_method=class_sample_method,
                            instance_sample_method=instance_sample_method,
                            num_of_db_instance=num_of_unseen_db_instance,
                            num_of_query_class=num_of_unseen_query_class,
                            num_of_query_instance_per_class=num_of_unseen_query_instance_per_class,
                            maximum_of_sampled_data=1000000)

        # Prepare total set
        # DB
        sampled_seen_db_instance_ids = sampled_seen[sample_fields.db_instance_ids]
        sampled_unseen_db_instance_ids = sampled_unseen[sample_fields.db_instance_ids]
        sampled_seen_db_label_ids = sampled_seen[sample_fields.db_label_ids]
        sampled_unseen_db_label_ids = sampled_unseen[sample_fields.db_label_ids]

        # Query
        sampled_seen_query_instance_ids = sampled_seen[sample_fields.query_instance_ids]
        sampled_unseen_query_instance_ids = sampled_unseen[sample_fields.query_instance_ids]
        sampled_seen_query_label_ids = sampled_seen[sample_fields.query_label_ids]
        sampled_unseen_query_label_ids = sampled_unseen[sample_fields.query_label_ids]

        # Embeddings
        sampled_seen_query_embeddings = embedding_container.get_embedding_by_instance_ids(
            sampled_seen_query_instance_ids)
        sampled_seen_db_embeddings = embedding_container.get_embedding_by_instance_ids(
            sampled_seen_db_instance_ids)
        sampled_unseen_query_embeddings = embedding_container.get_embedding_by_instance_ids(
            sampled_unseen_query_instance_ids)
        sampled_unseen_db_embeddings = embedding_container.get_embedding_by_instance_ids(
            sampled_unseen_db_instance_ids)


        total_db_instance_ids = sampled_seen_db_instance_ids + sampled_unseen_db_instance_ids
        total_db_label_ids = sampled_seen_db_label_ids + sampled_unseen_db_label_ids
        total_db_embeddings = embedding_container.get_embedding_by_instance_ids(total_db_instance_ids)
        print('shape of total db embeddings: {}'.format(total_db_embeddings.shape))

        print('[S] #of db instances: {} ({} classes), #of query instances: {} ({} classes)'.format(
            len(sampled_seen_db_label_ids), len(set(sampled_seen_db_label_ids)),
            len(sampled_seen_query_label_ids), len(set(sampled_seen_query_label_ids))))
        print('[U] #of db instances: {} ({} classes), #of query instances: {} ({} classes)'.format(
            len(sampled_unseen_db_label_ids), len(set(sampled_unseen_db_label_ids)),
            len(sampled_unseen_query_label_ids), len(set(sampled_unseen_query_label_ids))))
        print('[T] #of instances: {} ({} classes)'. format(
            len(total_db_label_ids), len(set(total_db_label_ids))))

        sampled_seen_query_label_ids = np.asarray(sampled_seen_query_label_ids)
        sampled_seen_db_label_ids = np.asarray(sampled_seen_db_label_ids)
        sampled_unseen_query_label_ids = np.asarray(sampled_unseen_query_label_ids)
        sampled_unseen_db_label_ids = np.asarray(sampled_unseen_db_label_ids)
        total_db_label_ids = np.asarray(total_db_label_ids)

        sampled_seen_query_instance_ids = np.asarray(sampled_seen_query_instance_ids)
        sampled_seen_db_instance_ids = np.asarray(sampled_seen_db_instance_ids)
        sampled_unseen_query_instance_ids = np.asarray(sampled_unseen_query_instance_ids)
        sampled_unseen_db_instance_ids = np.asarray(sampled_unseen_db_instance_ids)
        total_db_instance_ids = np.asarray(total_db_instance_ids)

        for _attr_name in self._attributes:
            #option configs
            print('Execute {} ranking evaluation:'.format(_attr_name))

            # Prepare instance ids, label ids and embeddings for different scenarios
            if _attr_name == attr_fields.seen_to_seen:
                """Seen To Seen"""
                print('#of instances: {}, # of class: {}'.format(
                    len(seen_instance_ids), len(set(seen_label_ids))))
                if len(seen_instance_ids) == 0 or len(set(seen_label_ids)) == 0:
                    print('Instances not enough, skip.')
                    continue

                query_embeddings = sampled_seen_query_embeddings
                query_label_ids = sampled_seen_query_label_ids
                query_instance_ids = sampled_seen_query_instance_ids
                db_embeddings = sampled_seen_db_embeddings
                db_label_ids = sampled_seen_db_label_ids
                db_instance_ids = sampled_seen_db_instance_ids

                print('# of sampled query: {}, db: {}'.format(
                    len(query_label_ids), len(db_label_ids)))
                if len(query_label_ids) == 0 or len(db_label_ids) == 0:
                    print('Data not enough, skip.')
                    continue

            elif _attr_name == attr_fields.unseen_to_unseen:
                """Unseen To Unseen"""
                print('{}: #of instances: {}, # of class: {}'.format(
                    _attr_name, len(unseen_instance_ids), len(set(unseen_label_ids))))
                if len(unseen_instance_ids) == 0 or len(set(unseen_label_ids)) == 0:
                    print('Instances not enough, skip.')
                    continue

                query_embeddings = sampled_unseen_query_embeddings
                query_label_ids = sampled_unseen_query_label_ids
                query_instance_ids = sampled_unseen_query_instance_ids
                db_embeddings = sampled_unseen_db_embeddings
                db_label_ids = sampled_unseen_db_label_ids
                db_instance_ids = sampled_unseen_db_instance_ids

                print('{}: # of sampled query: {}, db: {}'.format(
                    _attr_name, len(query_label_ids), len(db_label_ids)))
                if len(query_label_ids) == 0 or len(db_label_ids) == 0:
                    print('Data not enough, skip.')
                    continue

            else:
                if _attr_name == attr_fields.unseen_to_total:
                    """Unseen To Total"""
                    query_embeddings = sampled_unseen_query_embeddings
                    query_label_ids = sampled_unseen_query_label_ids
                    query_instance_ids = sampled_unseen_query_instance_ids
                    db_embeddings = total_db_embeddings
                    db_label_ids = total_db_label_ids
                    db_instance_ids = total_db_instance_ids

                    print('{}: # of sampled query: {}, db: {}'.format(
                        _attr_name, len(query_label_ids), len(db_label_ids)))
                    if len(query_label_ids) == 0 or len(db_label_ids) == 0:
                        print('Data not enough, skip.')
                        continue

                elif _attr_name == attr_fields.seen_to_total:
                    """Seen To Total"""
                    query_embeddings = sampled_seen_query_embeddings
                    query_label_ids = sampled_seen_query_label_ids
                    query_instance_ids = sampled_seen_query_instance_ids
                    db_embeddings = total_db_embeddings
                    db_label_ids = total_db_label_ids
                    db_instance_ids = total_db_instance_ids

                    print('{}: # of sampled query: {}, db: {}'.format(
                        _attr_name, len(query_label_ids), len(db_label_ids)))
                    if len(query_label_ids) == 0 or len(db_label_ids) == 0:
                        print('Data not enough, skip.')
                        continue

            """
            The Following Section should be a module.
            """
            print('{}: {} query instances with {} classes search on db {} instances with {} classes.'.format(
                _attr_name, len(query_embeddings), len(set(query_label_ids)), len(db_embeddings), len(set(db_label_ids))))
            metric_results = self._run_ranking(
                query_embeddings,
                query_label_ids,
                query_instance_ids,
                db_embeddings,
                db_label_ids,
                db_instance_ids,
                save_report)

            result_container.add(_attr_name, checkout_fields.top_k_hit_accuracy,
                                metric_results[1], condition={'k': 1})
            result_container.add(_attr_name, checkout_fields.top_k_hit_accuracy,
                                metric_results[5], condition={'k': 5})
            result_container.add(_attr_name, checkout_fields.mAP,
                                 metric_results[checkout_fields.mAP])

            if save_report is not None:
                report_table[_attr_name] = metric_results['instance_result']

        # Save table out.
        if save_report is not None:
            reporter = Reporter(report_table)
            reporter.save_report(save_report)

        return result_container

    def _run_ranking(self,
                     query_embeddings,
                     query_label_ids,
                     query_instance_ids,
                     db_embeddings,
                     db_label_ids,
                     db_instance_ids,
                     save_report=None,
                     ):

        # TODO @kv: Add more evaluation and reranking methods
        # NOTE: query labels are groundtruths
        rank_result = {}
        rank_result['instance_result'] = list()
        top_1, top_k = 1, 5
        ranking_metrics = RankingMetrics(top_k)
        hit_arrays = np.empty((query_embeddings.shape[0], top_k), dtype=np.bool)

        for _idx, (_query_embed, _query_label, _query_inst) in enumerate(
            zip(query_embeddings, query_label_ids, query_instance_ids)):
            # naive search
            distances = euclidean_distance(_query_embed, db_embeddings)
            indexed_db_labels = indexing_array(distances, db_label_ids)
            hits = indexed_db_labels[:top_k] == _query_label
            hit_arrays[_idx, ...] = hits
            if save_report is not None:
                indexed_db_instances = indexing_array(distances, db_instance_ids)
                # NOTE: RAW Results Exporter.
                rank_result['instance_result'].append({
                    'query_label_id': str(_query_label),
                    'query_instance_id': str(_query_inst),
                    'top_k_retrieval_label_ids': indexed_db_labels[:top_k].tolist(),
                    'top_k_retrieval_instance_ids': indexed_db_instances[:top_k].tolist(),
                    'is_hit': bool(hits[0]),
                })

        ranking_metrics.add_inputs(hit_arrays)
        rank_result[top_1] = ranking_metrics.top1_hit_accuracy
        rank_result[top_k] = ranking_metrics.topk_hit_accuracy
        rank_result[checkout_fields.mAP] = ranking_metrics.mean_average_precision

        return rank_result