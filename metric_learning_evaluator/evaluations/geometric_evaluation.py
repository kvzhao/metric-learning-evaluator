"""Geometric Evaluation

  Functionalities:
    - Euclidean Margin
    - Angular Margin
    - Entropy?
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from collections import Counter

from sklearn.cluster import AgglomerativeClustering
from metric_learning_evaluator.evaluations.evaluation_base import MetricEvaluationBase
from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.data_tools.result_container import ResultContainer

from metric_learning_evaluator.index.agent import IndexAgent


class GeometricEvaluation(MetricEvaluationBase):

    def __init__(self, config, mode=None):
        """
        """
        super(GeometricEvaluation, self).__init__(config, mode)

        print ('Create {}'.format(self.evaluation_name))
        self.show_configs()

    @property
    def metric_names(self):
        _metric_names = []
        for _metric_name, _content in self.metrics.items():
            for _attr_name in self.attributes:
                if _content is None:
                    continue
                _name = '{}/{}'.format(_attr_name, _metric_name)
                _metric_names.append(_name)
        return _metric_names

    def compute(self, container):
        """
          Args:
            container: EmbeddingContainer
          Return:
            res_container: ResultContainer
        """
        self.result_container = ResultContainer()

        for group_cmd in self.group_commands:
            instance_ids_given_attribute = \
                container.get_instance_id_by_group_command(group_cmd)
            if len(instance_ids_given_attribute) == 0:
                continue
            self._per_class_margin(group_cmd,
                instance_ids_given_attribute, container)


        return self.result_container

    def _per_class_margin(self, attr_name, instance_ids, container):
        """
        """
        agent_type = self.configs.agent_type

        embeddings = container.get_embedding_by_instance_ids(instance_ids)
        label_ids = container.get_label_by_instance_ids(instance_ids)

        # A map from instance_id to label_id
        #labelmap = {inst_id: label_id 
        #    for inst_id, label_id in zip(instance_ids, label_ids)}

        agent = IndexAgent(agent_type, instance_ids, embeddings)
        label_set = list(set(label_ids))

        print('total len of instance: {}, feat shape: {}'.format(len(instance_ids), embeddings.shape))
        # Goal: Analyze inter cluster & intra cluster

        def _distance_gaps(distances):
            gaps = distances[1:] - distances[:-1]
            return gaps

        # All class
        for cluster_label in label_set:
            outsider_instance_ids = container.get_instance_ids_by_exclusive_label(cluster_label)

            # Intra Cluster
            cluster_embeddings = container.get_embedding_by_label_ids(cluster_label)
            cluster_instance_ids = container.get_instance_ids_by_label(cluster_label)
            cluster_agent = IndexAgent(agent_type, cluster_instance_ids, cluster_embeddings)
            distance_matrix = cluster_agent.distance_matrix

            print('#cluster:{}(id={}), #outsider:{}'.format(
                len(cluster_instance_ids), cluster_label, len(outsider_instance_ids)))

            # per element inside the cluster
            """
            for elem_id, elem_feat in zip(cluster_instance_ids, cluster_embeddings):
                sorted_indices, distances = agent.search(elem_feat, 100)
                print(distances)
            """
            for elem_id, elem_feat in zip(cluster_instance_ids, cluster_embeddings):
                # TODO: add top_k to config
                retrieved_instances, retrieved_distances = agent.search(elem_feat, 100) # batch process
                retrieved_instances = np.squeeze(retrieved_instances)
                retrieved_distances = np.squeeze(retrieved_distances)
                retrieved_labels = np.asarray(container.get_label_by_instance_ids(retrieved_instances))

                hits = retrieved_labels == cluster_label
                hit_counter = Counter(retrieved_labels)
                hit_indices = np.where(hits == True)[0]

                gaps = _distance_gaps(retrieved_distances)
                sorted_gap_indices = np.argsort(gaps)[::-1]

                first_negative_index = np.where(hits == False)[0][0]
                last_positive_index = hit_indices[-1]

                print(first_negative_index, last_positive_index)
                if first_negative_index > last_positive_index:
                    print('Clean case')

                #plt.plot(retrieved_distances)
                #plt.stem(gaps, linefmt='--')
                #plt.savefig('figs/gap_{}.png'.format(elem_id))
                """
                #cluster:50(id=1), #outsider:16266
                [    1    21    25    30    10    41 11203     4 11027 11094 11197    11
                976   987  3372  3393 10985  4221  4212 10993  9056  5735 12032 11997
                4230   962  5734   960  5720  2973 10999    40   971 12411 11005 11226
                4157  5717 12430  4170  6636 11106  5726    46 11198 11009  9045 11018
                4237 11207  3399    19 10986 11014 15961  6648  4228   578 11028 11047
                11052  5749 11123 11051 12241 10989  7805 11021 11060  2998    29  9083
                6389 12437  6375 12708 16000 12711  5709   994 12417 12423 12429 12446
                9461 11208  5754  4064    48  2782   988  9402  9426  2991  6395  9062
                770  7833  9497 11067]
                1
                [  1   1   1   1   1   1 254   1 250 252 254   1  20  20  68  68 250  85
                85 250 204 116 273 273  85  20 116  20 116  60 250   1  20 282 250 254
                84 116 282  84 143 252 116   1 254 250 204 250  85 254  68   1 250 250
                371 143  85  12 251 251 251 116 252 251 277 250 169 250 251  60   1 204
                136 282 136 288 371 288 116  20 282 282 282 282 214 254 116  82   1  56
                20 213 213  60 136 204  16 169 214 251]
                [ True  True  True  True  True  True False  True False False False  True
                False False False False False False False False False False False False
                False False False False False False False  True False False False False
                False False False False False False False  True False False False False
                False False False  True False False False False False False False False
                False False False False False False False False False False  True False
                False False False False False False False False False False False False
                False False False False  True False False False False False False False
                False False False False]
                Counter({1: 13, 250: 11, 116: 8, 282: 7, 20: 7, 251: 6, 254: 6, 85: 5, 204: 4, 68: 3, 136: 3, 252: 3, 60: 3, 213: 2, 143: 2, 273: 2, 84: 2, 214: 2, 288: 2, 169: 2, 371: 2, 12: 1, 16: 1, 82: 1, 56: 1, 277: 1})
                hit indices [ 0  1  2  3  4  5  7 11 31 43 51 70 88]
                [0.         0.         0.         0.04722505 0.39277944 0.42956826
                0.6277188  0.70875573 0.7678457  0.7681575  0.77059966 0.77769154
                0.77795607 0.77795607 0.7876685  0.7876685  0.804518   0.8174874
                0.82676756 0.8278451  0.8348964  0.8362886  0.8385302  0.8394103
                0.8422515  0.8439253  0.8520241  0.8575696  0.8646956  0.8674249
                0.8676323  0.87297386 0.8733571  0.8809374  0.8833335  0.8877235
                0.8894118  0.8904713  0.89223945 0.89718133 0.8982529  0.9012508
                0.90286225 0.90706563 0.9078656  0.9105606  0.910897   0.91305894
                0.914123   0.9173614  0.9195067  0.9203303  0.9205645  0.9255354
                0.92966807 0.93422747 0.93455166 0.9348095  0.9381131  0.9381131
                0.9381131  0.9391134  0.9400259  0.94135803 0.9434733  0.9457501
                0.9458027  0.94778883 0.95063174 0.9506496  0.95319605 0.9552279
                0.95875996 0.95975405 0.9604337  0.96307033 0.9648634  0.96657157
                0.96732396 0.9675503  0.9675757  0.9675757  0.9675757  0.9675757
                0.97168714 0.9732246  0.9757464  0.97912467 0.98108554 0.9845027
                0.98504055 0.98851335 0.98851335 0.9893387  0.99049044 0.9908664
                0.9930371  0.993689   0.9946602  0.99589264]
                """
            #plt.savefig('figs/gap_{}.png'.format(elem_id))
            #plt.clf()
            break