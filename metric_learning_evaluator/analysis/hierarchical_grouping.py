"""HC Module
"""

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from metric_learning_evaluator.index.agent import IndexAgent
from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer


class HierarchicalGrouping(object):
    """
      Mode seperation according to embeddings.
    """

    def __init__(self, container):
        """
          Args:
            container: EmbeddingContainer
        """
        self._container = container
        self._internals()

    def _internals(self):
        # origin label id to new
        self._origin_table = self._container._instance_id_by_label
        self._subgroup_table = defaultdict(list)

    def label_subgroup(self, label_id, k_level=2):
        """
          Args:
            label_id,
            k_level: int
          Returns:
            sg_map:
                Dict of list. key is subgroup id and fellowing value is list of instance ids.
        """
        instance_given_label = self._container.get_instance_ids_by_label(label_id)
        num_of_instance = len(instance_given_label)
        if k_level > num_of_instance:
            print('NOTICE: Reduce split level {}->{}'.format(k_level, num_of_instance))
            k_level = min(k_level, num_of_instance)
        embedding_given_label = self._container.get_embedding_by_instance_ids(instance_given_label)
        subgroup_ids = AgglomerativeClustering(k_level).fit(embedding_given_label).labels_
        sg_map = defaultdict(list)
        for inst_id, sg_id in zip(instance_given_label, subgroup_ids):
            sg_map[sg_id].append(inst_id)
        return sg_map

    def auto_label_subgroup(self, label_id, use_dataframe=True):
        """
          Args:
            label_id: Integer
            use_dataframe: Boolean, set True by default
          Returns:
            hc_levels: Dataframe if `use_dataframe` is set true, otherwise the dictionary
        """
        label_name = self._container.labelmap[label_id]
        instance_given_label = self._container.get_instance_ids_by_label(label_id)
        embedding_given_label = self._container.get_embedding_by_instance_ids(instance_given_label)
        num_of_instance = len(instance_given_label)
        agent = IndexAgent(agent_type='HNSW',
                           instance_ids=instance_given_label,
                           embeddings=embedding_given_label)
        hc_levels = {}
        # All levels (20% of given instances)
        for k_level in range(2, num_of_instance // 5):
            subgroup_ids = AgglomerativeClustering(k_level).fit(embedding_given_label).labels_
            sg_map, id_map = defaultdict(list), {}
            # parse subgroup ids
            for sg_id, inst_id in zip(subgroup_ids, instance_given_label):
                sg_map[sg_id].append(inst_id)
                id_map[inst_id] = sg_id
            # compute centers
            sg_centers, centers = {}, []
            for sg_id, inst_ids in sg_map.items():
                sg_embeddings = self._container.get_embedding_by_instance_ids(inst_ids)
                sg_center = np.mean(sg_embeddings, axis=0)
                centers.append(sg_center)
                sg_centers[sg_id] = sg_center
            # analyze
            centers = np.asarray(centers)
            impurity_count = 0
            mean_cluster_size = 0.0
            k_level_instance_ids, k_level_filenames = [], []
            for sg_id, center in sg_centers.items():
                inst_ids = sg_map[sg_id]
                k_level_instance_ids.append(inst_ids)
                k_level_filenames.append(self._container.get_filename_strings_by_instance_ids(inst_ids))
                num_sg_elements = len(inst_ids)
                sg_embeddings = self._container.get_embedding_by_instance_ids(inst_ids)
                sg_rmax = np.sum((sg_embeddings - center)**2, axis=1)
                sg_rmax = np.max(sg_rmax)

                truncated_retrieved_ids, _ = agent.search(center, top_k=num_sg_elements)
                truncated_retrieved_ids = truncated_retrieved_ids[0].tolist()
                truncated_retrieved_sg_ids = np.asarray([id_map[rid] for rid in truncated_retrieved_ids])
                truncated_hit_array = truncated_retrieved_sg_ids == sg_id

                # to up last positive
                full_retrieved_ids, _ = agent.search(center, top_k=num_of_instance)

                # output
                impurity_count += (num_sg_elements - np.sum(truncated_hit_array))
                cluster_area = sg_rmax ** 2
                mean_cluster_size += cluster_area
            mean_cluster_size /= k_level
            hc_levels[k_level] = {
                'label_id': label_id,
                'label_name': label_name,
                'impurity_count': impurity_count,
                'impurity_ratio': impurity_count / num_of_instance,
                'mean_cluster_size': mean_cluster_size,
                'subgroup_instance_ids': k_level_instance_ids,
                'subgroup_filename_strings': k_level_filenames,
            }
        if not use_dataframe:
            return hc_levels
        return pd.DataFrame.from_dict(hc_levels, orient='index')

    def agnostic_subgroup(self):
        pass

    def auto_agnostic_subgroup(self):
        pass