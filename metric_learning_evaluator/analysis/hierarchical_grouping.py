"""Hierarchical Clustering Module
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

    def label_subgroup(self, label_id, depth_ratio=0.2, use_dataframe=True, pruned=None):
        """
          Args:
            label_id: Integer
            use_dataframe: Boolean, set True by default
            depth_ratio: float. Take percent of given instances as clustering level.
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
        depth = int(num_of_instance * depth_ratio)
        for k_level in range(2, depth):
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
            cluster_size={}
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

                # to up last positive (TODO: compute full purity)
                # full_retrieved_ids, _ = agent.search(center, top_k=num_of_instance)

                # output
                impurity_count += (num_sg_elements - np.sum(truncated_hit_array))
                cluster_area = sg_rmax  
                cluster_size[sg_id] = cluster_area
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

    def auto_label_subgroup(self, label_id, depth_ratio=0.2, use_dataframe=True, manual_k=None):
        """
          Args:
            label_id: Integer
            use_dataframe: Boolean, set True by default
          Returns:
            k_level: Integer
            sg_info: Dataframe if `use_dataframe` is set true, otherwise the dictionary
        """
        # k_level => first local minima location of impurity_count
        if not manual_k: 
           df = self.label_subgroup(label_id, depth_ratio, True)
           impurity_count = np.array(df.impurity_count)
           local_minima_loc = np.r_[True, impurity_count[1:] < impurity_count[:-1]] & np.r_[impurity_count[:-1] <= impurity_count[1:], True]
           k_level = np.array(df.index)[local_minima_loc][0]
           #print(argrelextrema(impurity_count, np.less))
        else:
           assert isinstance(manual_k, int), 'k-level should be an integer'
           k_level = manual_k
           
        print('Do hierarchical clustering on label {} with k_level {}'.format(label_id,k_level))

        # get instance id
        instance_given_label = self._container.get_instance_ids_by_label(label_id)
        embedding_given_label = self._container.get_embedding_by_instance_ids(instance_given_label)

        subgroup_ids = AgglomerativeClustering(k_level).fit(embedding_given_label).labels_
        sg_map, id_map = defaultdict(list), {}

        for sg_id, inst_id in zip(subgroup_ids, instance_given_label):
                sg_map[sg_id].append(inst_id)
                id_map[inst_id] = sg_id

        # compute centers & distance between center and surroundings
        sg_centers = {}
        center_path={}
        avg_distance={}
        sg_surroundings_distance={}
        sg_surroundings_path={}
        sg_surroundings_embeds={}
        
        for sg_id, inst_ids in sg_map.items():
            sg_embeddings = self._container.get_embedding_by_instance_ids(inst_ids)
            sg_center = np.mean(sg_embeddings, axis=0)
            sg_centers[sg_id] = sg_center
            
            embed_distance = np.sum((sg_embeddings - sg_center)**2, axis=1)
            embed_distance_sort = np.sort(embed_distance)
            avg_distance[sg_id] = np.sum(embed_distance_sort)/len(embed_distance_sort)

            # get the info of surroundings from the near to the distant
            surroundings=[]
            for i in embed_distance_sort:
               loc = np.where(embed_distance == i)[0][0]
               surrounding = inst_ids[loc] 
               surroundings.append(surrounding)

            sg_surroundings_path[sg_id] = self._container.get_filename_strings_by_instance_ids(surroundings)
            sg_surroundings_distance[sg_id] = embed_distance_sort
            #sg_surroundings_embeds[sg_id] = self._container.get_embedding_by_instance_ids(surroundings)

        sg_info = {}
        for sg_id in sg_map.keys():
            sg_info[sg_id] = {
             'counts': len(sg_map[sg_id]),
             'sg_surroundings_path': sg_surroundings_path[sg_id],
             'sg_surroundings_distance':sg_surroundings_distance[sg_id],
             'avg_distance': avg_distance[sg_id]
        }
        if not use_dataframe:
             return sg_info
        return k_level, pd.DataFrame.from_dict(sg_info, orient='index')

