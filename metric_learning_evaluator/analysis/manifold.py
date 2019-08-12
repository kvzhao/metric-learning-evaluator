import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from tqdm import tqdm
from metric_learning_evaluator.index.agent import IndexAgent

from metric_learning_evaluator.index.utils import euclidean_distance
from metric_learning_evaluator.index.utils import angular_distance
from metric_learning_evaluator.index.utils import indexing_array

from metric_learning_evaluator.metrics.ranking_metrics import RankingMetrics
from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.utils.switcher import switch


class Cluster(object):
    """Cluster is a group of same label embeddings.
    """

    def __init__(self, label_id, embeddings):
        pass

    @property
    def center(self):
        pass

    @property
    def cluster_size(self):
        pass

# TODO (kv): WOW! this needs refactoring!
class Manifold(object):
    """Manifold is the geometric analysis tool.
       It describes the global geometric point of view and remain the relation.

      Several options can be considerred:
       - distance trace
       - margin
       - locality
       - global structure
       - local structure
    """
    def __init__(self, embedding_container, label_names=None, agent_type='HNSW'):
        """
          Args:
            embedding_container:
                Object of EmbeddingContainer
            label_names:
                Easier for human understanding
            index_agent: String, option: HWNS | Numpy
        """
        self._label_names = label_names

        # init data structure
        self.clear()

        # process fundamental relations
        self._embedding_container = embedding_container
        self._preprocess()

        # search engine
        self._agent = IndexAgent(agent_type,
            self._embedding_container.instance_ids,
            self._embedding_container.embeddings)
        print('Manifold is initialized')

    def _create_labelmap(self, label_ids, label_names):
        """
          Args:
            label_ids: list or array of integers
            label_names: list or array of strings
              both label_ids & label_names are in same order.
        """
        labelmap = {}
        for label_id, name in zip(label_ids, label_names):
            if not label_id in labelmap:
                labelmap[label_id] = name
            else:
                # check consistency
                if labelmap[label_id] != name:
                    print('WARNING: {}:{} not consistent with {}'.format(
                        label_id, labelmap[label_id], name))
        return labelmap

    def _preprocess(self):
        """Preprocess
            build up relations and indices of each label & instance id
            1. class_ids
            2. class_id to {instance_ids}
            3. class_id to class_name
        """
        # specs
        # turn into set
        self._class_ids = list(set(self._embedding_container.label_ids))
        print('container: {} instances with {} classes'.format(
            len(self._embedding_container.label_ids), len(self._class_ids)))

        # class_id: [instance_ids]
        for _class_id in self._class_ids:
            _class_id = int(_class_id)
            same_class_instance_ids = self._embedding_container.get_instance_ids_by_label(_class_id)
            self._class_to_instance_ids[_class_id] = np.asarray(same_class_instance_ids)

        if self._label_names is not None:
            self._labelmap = self._create_labelmap(
                self._embedding_container.label_ids, self._label_names)
            print('label names are loaded.')

    def class_center(self):
        """Get mean feature of each given class ids
          Return:
            centers: dict, map class_id to class_center.
        """

        def _compute_center(same_class_instance_ids):
            same_class_embeddings = self._embedding_container.get_embedding_by_instance_ids(same_class_instance_ids)
            same_class_center = np.mean(same_class_embeddings, axis=0)
            return same_class_center

        all_center_embeddings = []
        all_center_ids = []
        for _class_id in self._class_ids:
            _same_class_instances = self._class_to_instance_ids[_class_id]
            center = _compute_center(_same_class_instances)
            self._all_center[_class_id] = center
            all_center_ids.append(_class_id)
            all_center_embeddings.append(center)

        self._all_center_embeddings = np.asarray(all_center_embeddings)
        self._all_center_ids = np.asarray(all_center_ids)
        return self._all_center 

    @staticmethod
    def _find_large_gap(distance_ray, top_k=None):
        """
          Args:
            distance_ray: 1D array
            threshold: a float to define `large`, return all if None is given.
          Returns:
            (sorted_indices, sorted_gaps)
              sorted_gaps: numpy array in decreasing order
              sorted_indices: numpy array of indices
        """
        dist_gap = distance_ray[1:] - distance_ray[:-1]
        sorted_indices = dist_gap.argsort()[::-1]
        sorted_gaps = dist_gap[sorted_indices]
        return sorted_indices[:top_k], sorted_gaps[:top_k]

    @staticmethod
    def _ranking_pattern(query_label, ranked_labels, distance):
        """Recognize ranking pattern by observing is same or not
           -> mAP actually.

          Args:
            query_label:
            ranked_labels:
            distance: 
          Retrun:
            pattern:
                dict
        """
        pattern = None
        hit_array = ranked_labels == query_label

        first_negative_index = np.argwhere(hit_array == False)
        last_positive_index = first_negative_index - 1

        return pattern


    # --> geometric analysis
    def distance_trace(self, query_label_id, query_embedding, top_k=2000):
        """Distance trace: A ray through ranked distances.

          Args:
            query_embedding:
            query_label_id:
          Return:
            results: Dictionary of results
                - 
        """

        indices, distances = self._agent.search(query_embedding, top_k)
        indices = np.squeeze(indices)
        distances = np.squeeze(distances)
        retrieved_label_ids = self._embedding_container.get_label_by_instance_ids(indices)

    def locality_analysis(self):
        """Locality measures variation between each center and components. 
        """
        if not self._all_center:
            self.class_center()

        def _search(query, database, indices):
            distances = euclidean_distance(query, database)
            sorted_distances = indexing_array(distances, distances)
            sorted_indices = indexing_array(distances, indices)
            return sorted_indices, sorted_distances

        def _cluster_size(sorted_distances):

            if len(sorted_distances) > 1:
                return sorted_distances[-1] + sorted_distances[-2]
            elif len(sorted_distances) == 1:
                return sorted_distances[0]
            else:
                return 0.0

        def _find_outlier(sorted_distances, threshold):
            deviation = np.std(sorted_distances)
            medium = np.median(sorted_distances)
            pass

        for _class_id, _instance_ids in self._class_to_instance_ids.items():
            cluster_embeddings = self._embedding_container.get_embedding_by_instance_ids(_instance_ids)
            center_embedding = self._all_center[_class_id]
            sorted_instance_ids, sorted_distances = _search(center_embedding, cluster_embeddings, _instance_ids)
            cluster_size = _cluster_size(sorted_distances)
            print(cluster_size)

    def global_analysis(self):
        pass


    def center_to_center_relation(self):
        """Center to center distance matrix
          Return:
            c2c_matrix: CxC, 2d array
        """
        size = self._all_center_embeddings.shape[0]
        c2c_matrix = np.empty((size, size))

        for _idx, _query in enumerate(self._all_center_embeddings):
            c2c_matrix[_idx, ...] = euclidean_distance(_query, self._all_center_embeddings)

        return c2c_matrix

    def center_to_all_instance_relation(self, top_k=2000):
        """Center to All instances distance relation. 
          Return:
            c2all_matrix: A 2d matrix with shape (num_center, top_k). The order of centers
                          is same with self._all_center_ids.
        """
        c2all_indices, c2all_distances = self._agent.search(self._all_center_embeddings, top_k)
        return c2all_distances

    def one_class_pairwise_relation(self, label_id, pair_num_limit=None):
        """
          Args:
            label_id : 
            pair_num_limit : 
          Return:
            intra_class_angles: 1D numpy, angles between all pairs that have label_id,
                                 size < pair_num_limit if given
            inter_class_angles : 1D numpy, angles between randomly selected negative pair,
                                 size < pair_num_limit if given
        """
        def _randomly_select_negative_pair_instance_ids(label_id, num_pairs):
            random_indices = []
            for _ in range(num_pairs):
                random_idx = np.random.randint(0, len(self._embedding_container.instance_ids))
                if self._embedding_container.get_label_by_instance_ids(random_idx) != label_id:
                    random_indices.append(random_idx)
            return random_indices

        embeddings = self._embedding_container.get_embedding_by_label_ids(label_id)
        num_instances = len(embeddings)

        intra_class_angles = []
        for i in range(num_instances):
            for j in range(i+1, num_instances):
                if pair_num_limit is not None and len(intra_class_angles) >= pair_num_limit:
                    break
                intra_class_angles.append(angular_distance(embeddings[i], embeddings[j]))

        negative_embeddings = self._embedding_container.get_embedding_by_instance_ids(
            _randomly_select_negative_pair_instance_ids(label_id, len(intra_class_angles)) )

        inter_class_angles = []
        for u, v in zip(embeddings, negative_embeddings):
            inter_class_angles.append(angular_distance(u,v))

        return np.array(intra_class_angles), np.array(inter_class_angles)

    def all_pairwise_relation(self, pair_num_limit=None):
        """ Call one_class_pair_wise_relation for all label ids """
        all_intra_angles = []
        all_inter_angles = []
        for label_id in tqdm(self._labelmap.keys()):
            # TODO : change the type in embedding container
            if not isinstance(label_id, int):
                label_id = label_id.item() # np.int16, 32, 64.. --> python int
            
            intra_angles, inter_angles = self.one_class_pairwise_relation(label_id, pair_num_limit)
            all_intra_angles.extend(intra_angles)
            all_inter_angles.extend(inter_angles)
        return np.array(all_intra_angles), np.array(all_inter_angles)

    def clear(self):
        self._labelmap = None
        self._class_ids = None

        self._all_center_embeddings = None
        self._all_center_ids = None

        self._all_center = {}
        self._class_to_instance_ids = {}

class CrossManifoldAnalysis(object):
    pass
