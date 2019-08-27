"""Auto Verification Kernel

  Deployment versions:
    - Evaluator version == ?
    - DB Verification version == ?
    **Not yet fork release branch

  Interfaces:
    - load
        Load each data from disk
        - load_embedding
        - load_result
        - load_joined_df
    - restore
        Load and check consistency, no need to execute preprocessing
    - save
        Use auto_save by default
    - preprocessing

    - run

  Configs:
    - sys_config
    - run_config

  WORKING PROCEDURE
   - load: Convert EmbeddingDB to EmbeddingContainer (load_pkl)
   - preprocessing: Execute Index Agent
   - _save_result: Save indices & distances into ResultContainer
   - Send EmbeddingContainer & ResultContainer both into analysis process


  TODO:
    - function status (ok)
    - system status (ready)

    Authors: kv, lotus
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import yaml

import numpy as np
import pandas as pd
from functools import reduce

# Evaluator
from metric_learning_evaluator.index.agent import IndexAgent
from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.data_tools.result_container import ResultContainer
from metric_learning_evaluator.core.standard_fields import AutoVerificationStandardFields as Fields

# DB Verification
from db_verification.measure import hit
from db_verification.measure import purity
from db_verification.measure import rerank
from db_verification.measure import accuracy
from db_verification.measure import n_switch
from db_verification.measure import k_fold_margin
from db_verification.measure import count_hit_by_class
from db_verification.measure import per_class_top_k_hit
from db_verification.measure import per_class_top_k_accuracy
from db_verification.measure import roc_per_category_margin
from db_verification.measure import wasserstein_distribution_distance

from db_verification.utils import join
from db_verification.utils import rank
from db_verification.utils import correct
from db_verification.utils import filter_by_domain


def parse_sys_config(config_dict=None):
    pass

def parse_run_config(config_dict=None):
    pass


class AutoVerification(object):
    def __init__(self, config_dict=None, verbose=False):
        self._config_dict = config_dict
        self._verbose = verbose

        # create folder & env info
        self._setup_env()

        # init container, result & agent
        self._setup_internals()

        # minimum dataframe records embedding & results
        self._setup_dfs()

        # READY Definition: result container is computed
        self._ready = False

    def _setup_env(self):
        """Setup workspace and parse sys_config
        """
        if Fields.workspace_path not in self._config_dict:
            self._config_dict[Fields.workspace_path] = '/tmp/auto_verification_workspace'
        workspace_config = self._config_dict[Fields.workspace_path]

        self._root_path = workspace_config[Fields.root_path]
        if not os.path.exists(self._root_path):
            os.makedirs(self._root_path)

        if Fields.result_container_name in workspace_config and workspace_config[Fields.result_container_name]:
            self._result_container_path = os.path.join(self._root_path,
                                                       workspace_config[Fields.result_container_name])

        if Fields.joined_df_name in workspace_config and workspace_config[Fields.joined_df_name]:
            self._joined_df_path = os.path.join(self._root_path,
                                                workspace_config[Fields.joined_df_name])

        if Fields.truncation not in self._config_dict:
            self._truncation = None
        else:
            self._truncation = self._config_dict[Fields.truncation]

    def _setup_dfs(self):
        # Fundamentals
        self._joined_df = None

        self._query_df = None
        self._anchor_df = None

        self._q2g_df = None
        self._g2q_df = None

        self._final_report = None

    def _setup_internals(self):
        self._container = None
        self._result = None
        self._agent = None

    def load_embedding_db(self, pkl_path):
        """Load Cradle.EmbeddingDB format pkl file.
          Args:
           pkl_path: String, path to .pkl exported by cradle
           NOTE: md5 would not be checked.
          Raise:
            NONE
        """
        if self._container is None:
            self._container = EmbeddingContainer()
            self._container.load_pkl(pkl_path)
        else:
            if self._verbose:
                print('NOTICE: EmbeddingContainer already exist, reload from {}'.format(pkl_path))
            self._container.clear()
            self._container.load_pkl(pkl_path)
            # Back to not ready (really?)
            self.check_ready()

        if self._verbose:
            print(self._container)

    def load_embedding_container(self, embedding_dir):
        """Load metric_learning_evaluator EmbeddingContainer folder.
          Args:
           embedding_dir: String, path to folder exported by embedding container.
        """
        if self._container is None:
            self._container = EmbeddingContainer()
            self._container.load(embedding_dir)
        else:
            if self._verbose:
                print('NOTICE: EmbeddingContainer already exist, reload from {}'.format(embedding_dir))
            self._container.clear()
            self._container.load(embedding_dir)

        if self._verbose:
            print(self._container)

    def load_result_container(self, result_dir):
        """Load metric_learning_evaluator ResultContainer folder.
          Args:
           result_dir: String, path to folder exported by result container.
        """
        if self._result is None:
            self._result = ResultContainer()
        # check files are under the given folder
        if result_dir is not None and os.path.exists(result_dir):
            self._result.load(result_dir)
        self.check_ready()

    def save_result_container(self, result_dir):
        if self._result is None or not self.ready:
            return
        self._result.save(result_dir)

    def save_joined_df(self, dataframe_path):
        if self._joined_df is None or not self.ready:
            return
        self._joined_df.to_pickle(dataframe_path + '.pkl')

    def load_joined_df(self, dataframe_path):
        """Load joined dataframe with pickle format"""
        # TODO: Check status, notice if exist
        self._joined_df.read_pickle(dataframe_path + '.pkl')

    def load(self, embedding_pkl_path, result_container_path=None, joined_df_path=None):
        """
          Args:
            embedding_pkl_path: string, path to embedding_db pkl file
            result_container_path: string, path to result container folder
            joined_df_path: string, path to saved dataframe from pkl file

          Procedure:
            - Use given args first if given
            - Check sys_config's path if result & joined dataframe exist
          TODO:
            logic: if result is given but df not (or not consistent!?), execute join
        """
        self.load_embedding_db(embedding_pkl_path)

        if result_container_path:
            self.load_result_container(result_container_path)
            self._join_indexes_df()
        else:
            # Check whether system path exists
            pass

        if joined_df_path:
            self.load_joined_df(joined_df_path)
        else:
            # Check whether system path exists
            pass

    def save(self):
        pass

    def preprocessing(self, force_update=False, save=True):
        """Execute indexing and save to result container, join both dataframes.
          Args:
            force_update: Boolean, if set True, compute indexing no matter result container exists or not.
            save: Boolean
          Raise:
            - No embedding container or embedding db is given
          NOTE:
            auto_save?
        """

        if self._container is None:
            raise ValueError('EmbeddingContainer or EmbeddingDB are not given')

        if not self.ready or force_update:
            # TODO: @kv more flexible `type.query` config
            query_command = Fields.query
            anchor_command = Fields.anchor

            self._execute_searching(query_command,
                                    anchor_command,
                                    self._truncation)

            # TODO: Move these two lines into update
            # Join two dataframes
            self._join_indexes_df()

            # Preprocessing dataframes
            self._measure_preprocessing()

            # TODO: @kv output save_dir
            if save:
                self.save_result_container(self._result_container_path)
                self.save_joined_df(self._joined_df_path)
        else:
            if self._verbose:
                print('EmbeddingContainer & ResultContainer are given, ready for analysis')

    def _execute_searching(self, query_command, anchor_command, truncation=None):
        """
          Args
        """
        # TODO: @kv flexible commands

        command = '{}->{}'.format(query_command, anchor_command)

        query_ids, anchor_ids = self._container.get_instance_id_by_cross_reference_command(command)
        query_embeddings = self._container.get_embedding_by_instance_ids(query_ids)
        anchor_embeddings = self._container.get_embedding_by_instance_ids(anchor_ids)

        num_of_anchor = anchor_embeddings.shape[0]
        num_of_query = query_embeddings.shape[0]

        if truncation is not None:
            if isinstance(truncation, int):
                num_of_retrivied = truncation
        else:
            num_of_retrivied = num_of_anchor

        if self._verbose:
            print('{} anchors, {} queries'.format(num_of_anchor, num_of_query))

        self._agent = IndexAgent(agent_type='HNSW',
                                 instance_ids=anchor_ids,
                                 embeddings=anchor_embeddings)

        if self._verbose:
            print('Start indexing...')
        all_query_ids, all_retrieved_ids, all_retrieved_distances = [], [], []
        for _idx, (query_id, query_emb) in enumerate(zip(query_ids, query_embeddings)):
            # TODO: Add truncation option
            retrieved_ids, retrieved_distances = self._agent.search(query_emb, top_k=num_of_retrivied)
            retrieved_ids = np.squeeze(retrieved_ids)
            retrieved_distances = np.squeeze(retrieved_distances)

            all_query_ids.extend(np.array(query_id).repeat(num_of_retrivied))
            all_retrieved_ids.extend(retrieved_ids)
            all_retrieved_distances.extend(retrieved_distances)

        if self._verbose:
            print('Indexing finished, {} retrieved events'.format(len(all_retrieved_ids)))
            print('Start exporting results...')

        if self._result is None:
            self._result = ResultContainer()
        self._result._event_buffer = pd.DataFrame(
            {
                Fields.query_instance_id: all_query_ids,
                Fields.retrieved_instance_id: all_retrieved_ids,
                Fields.retrieved_distance: all_retrieved_distances,
            })
        if self._verbose:
            print('Preprocessing done')

        self.ready = True

    def _join_indexes_df(self):
        """Join embeddings and result events
        """
        # TODO: Check ready or not.
        if self._verbose:
            print('Start join two data frames...')
        container_df = self._container.DataFrame
        result_df = self._result.events
        self._joined_df = join(container_df, result_df)
        if self._verbose:
            print('dataframe joined')

    def _measure_preprocessing(self):
        if self._verbose:
            print('Start split dataframes into query & anchor...')
        query_cmd = '{}==\'{}\''.format(Fields.type, Fields.query)
        anchor_cmd = '{}==\'{}\''.format(Fields.type, Fields.anchor)
        self._query_df = self._container.DataFrame.query(query_cmd)
        self._gallery_df = self._container.DataFrame.query(anchor_cmd)

        if self._verbose:
            print('query_df & anchor_df are splitted')

        # Filter results: get `query` -> `gallery`
        self._q2g_df = filter_by_domain(self._joined_df,
                                        Fields.query,
                                        Fields.anchor)
        self._q2g_df = rank(self._q2g_df)
        self._q2g_df = correct(self._q2g_df)

        # TODO: g2q_df

    def run(self, run_config=None):
        """Calculate measures according to given run_config.

          Args:
            run_config: dict or None?
          Return:
            TODO: Do not return
            report: pandas.DataFrame, use run_config to control contents
          NOTE:
            run anchor_selection by default
            computed report would be stored inside object
            - must rerun when called
        """
        if not self.measure_preprocessed:
            self._measure_preprocessing()

        # Minimum settings: Do nothing
        if run_config is None or not run_config:
            # TODO: run basic performance check
            # Top k accuracy (category level)
            per_class_measures = [per_class_top_k_accuracy(self._q2g_df, 1),
                                  per_class_top_k_accuracy(self._q2g_df, 3),
                                  per_class_top_k_accuracy(self._q2g_df, 5)]
            # merge several dataframes
            # https://stackoverflow.com/questions/23668427/pandas-three-way-joining-multiple-dataframes-on-columns
            report = reduce(lambda left, right: pd.merge(left, right, on=Fields.query_label_name),
                            per_class_measures)
            self._final_report = report.rename_axis(Fields.label_name).reset_index()
            # Done.
            return

        # Perform measures

        # return empty

    def anchor_selection(self, select_config=None):
        """
          Args:
            select_config: dict
          Return:
            chosen_indices: list of instance_ids (int)
        """
        if self._final_report is None:
            # TODO: Call basic .run rather than return
            return []

        # Basic version: return all instance_ids
        if select_config is None or not select_config:
            ids = self._container.DataFrame.query(
                '{}==\'{}\''.format(Fields.type, Fields.anchor))[Fields.instance_id].tolist()
            return list(set(ids))

        return []

    def report(self, report_config=None, to_dict=False):
        """Generate auto verification report

            The function will decide scale of report
                - Overall
                - Category level
                - Instance level
        """
        if self._final_report is None:
            # TODO: What should I do?
            return {} if to_dict else pd.DataFrame()

        # Basic version, directly return
        if report_config is None or not report_config:
            if to_dict:
                return self._final_report.to_dict('list')
            else:
                return self._final_report

        # Parse report to different scale

        return {} if to_dict else pd.DataFrame()

    @property
    def ready(self):
        return self._ready

    @ready.setter
    def ready(self, is_ready):
        if isinstance(is_ready, bool):
            self._ready = is_ready

    @property
    def measure_preprocessed(self):
        """Check dataframes empty or not preprocessing needed?
        """
        if self._joined_df is None:
            return False
        if self._anchor_df is None or self._query_df is None:
            return False
        return True

    def check_ready(self):
        """Check the analysis (.run) is ready to perform.

          Ready condition:
            1. EmbeddingContainer is given
            2. ResultContainer is given
           *3. Indices in both containers are consistent
        """
        if self._container is None or self._result is None:
            self.ready = False
            return
        if self._result.events.empty:
            self.ready = False
            return
        self.ready = True

    def clear(self):
        if self._container:
            self._container.clear()
        if self._result:
            self._result.clear()
