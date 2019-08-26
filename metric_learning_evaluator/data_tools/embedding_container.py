"""EmbeddingContainer

    Define data containers for the metric learning evaluator.

    Brief intro:
        EmbeddingContainer:
            Efficient object which handles the shared (globally) embedding vectors.

    @bird, dennis, kv, lotus
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import re
import pickle
import numpy as np
import pandas as pd

from abc import ABCMeta
from abc import abstractmethod
from collections import defaultdict
from metric_learning_evaluator.data_tools.feature_object import FeatureObject
from metric_learning_evaluator.data_tools.attribute_table import AttributeTable
from metric_learning_evaluator.query.csv_reader import CsvReader

from metric_learning_evaluator.utils.interpreter import Interpreter
from metric_learning_evaluator.utils.interpreter import InstructionSymbolTable
from metric_learning_evaluator.core.standard_fields import AttributeStandardFields as attr_field
from metric_learning_evaluator.core.standard_fields import InterpreterStandardField as interpreter_field
from metric_learning_evaluator.core.standard_fields import EmbeddingContainerStandardFields as container_fields


class EmbeddingContainer(object):
    """The Data Container for Embeddings & Probabilities

      Container operanios:
        - add: an unique interface for adding datum into container
        - clear: clear the internal buffer
        - save:
        - load:
        - load_pkl:

      Query methods:

      = NOTE =======================================================================================
      NOTE: We CAN NOT confirm the orderness of logits & embedding consistent with instance_ids.
      TODO @kv: Use pandas dataframe as for query
      TODO @kv: Error-handling when current exceeds container_size
      TODO @kv: instance_id can be `int` or `filename`, this is ambiguous
      TODO @kv: smooth clear
      ==============================================================================================
    """

    def __init__(self,
                 embedding_size=0,
                 probability_size=0,
                 container_size=10000,
                 name='embedding_container'):
        """Constructor of the Container.
          Args:
            embedding_size: An integer,
                Dimension of the embedding vector, e.g. 128, 1024 or 2048 etc.
            probability_size: An integer:
                Disable this by giving size equals to 0.
            container_size: An integer,
                Number of embedding vector that container can store.
            name: A string
                The name string is used for version control.
        """
        assert embedding_size >= 0, 'embedding_size must larger than 0'
        assert container_size >= 0, 'container_size must larger than 0'
        assert probability_size >= 0, 'probability_size must larger than or equal to 0'
        self._embedding_size = embedding_size
        self._probability_size = probability_size
        self._container_size = container_size
        self._embeddings = None
        self._probabilities = None
        self._index_df = None
        self._name = name

        self._init_internals()
        self._init_arrays(self._container_size,
                          self._embedding_size,
                          self._probability_size)

        self._attribute_table = AttributeTable()
        self._interpreter = Interpreter()

    def __repr__(self):
        _content = '=' * 15 + ' {} '.format(self._name) + '=' * 15 + '\n'
        _content += 'embeddings: ({}, {})\n'.format(self.counts, self.embedding_size)
        if self._probabilities is not None:
            _content += 'probabilities: ({}, {})\n'.format(self.counts, self.probability_size)
        _content += 'internals: '
        if self._label_ids:
            _content += 'label_ids, '
        if self._label_names:
            _content += 'label_names, '
        if self._filename_strings:
            _content += 'filename_strings, '
        if self.attributes:
            _content += '\nattributes: {}'.format(', '.join(self.attributes))
        _content += '\n'
        _content += '=' * 50 + '\n'
        return _content

    def _init_internals(self):
        """Internals & Indexes
          Internal data
            - instance ids, label ids, label names, filename strings
            - Would be exported to .npy (feature_object) or .pkl (cradle EmbeddingDB)
          Indexing relations
        """
        # orderness should be maintained in _instance_ids
        self._instance_ids = []
        self._label_ids = []
        self._label_names = []
        self._filename_strings = []
        # init member data for indexing relations
        self._init_index_buffer()

    def _init_index_buffer(self):
        """Initialize Indexes"""
        # maps index used in numpy array and instance_id list
        self._index_by_instance_id = {}
        self._label_by_instance_id = {}
        self._label_name_by_instance_id = {}
        # instance_ids with same attribute
        # attribute-id mapping, shallow key-value pair
        self._instance_id_by_label = defaultdict(list)
        self._instance_by_attribute = {}
        self._attribute_by_instance = defaultdict(list)

    def _fetch_internals(self):
        """Fetch the dictionary of internal lists"""
        _internal_dict = {
            container_fields.instance_ids: self._instance_ids,
            container_fields.label_ids: self._label_ids,
            container_fields.label_names: self._label_names,
            container_fields.filename_strings: self._filename_strings}
        return _internal_dict

    def _init_arrays(self, container_size, embedding_size, probability_size):
        """Internal numpy arrays and array_index"""
        if container_size != self._container_size:
            self._container_size = max(self._container_size, container_size)
        if embedding_size != self._embedding_size:
            self._embedding_size = max(self._embedding_size, embedding_size)
        if probability_size != self._probability_size:
            self._probability_size = max(self._probability_size, probability_size)
        self._embeddings = np.empty((self._container_size,
                                     self._embedding_size), dtype=np.float32)
        if self._probability_size != 0:
            self._probabilities = np.empty((self._container_size,
                                            self._probability_size), dtype=np.float32)
        self._current = 0

    def add(self, instance_id, label_id, embedding,
            probability=None, attributes=None, label_name=None, filename=None):
        """Add datum interface for instance_id, label_id and embeddings.

          Args:
            instance_id: int
                Unique instance_id which can not be repeated in the container.
            label_id: int
                Index of given class corresponds to the instance.
            embedding: 1D numpy array,
                One dimensional embedding vector with size less than self._embedding_size.
            probability: 1D numpy array,
                One dimensional vector which records class-wise scores.
            attributes: A dictionary with attribute_name: attribute_value
            label_name: String
                Human-realizable content of given label_id
            filename: String
                The filename or filepath to the given instance_id.
          NOTICE:
            This should be the only interface data are added into container.
        """
        try:
            label_id = int(label_id)
            instance_id = int(instance_id)
        except:
            raise TypeError("The label id or instance id has wrong type")

        # assertions: embedding size
        assert embedding.shape[0] <= self._embedding_size, "Size of embedding vector is greater than the default."
        # TODO @kv: Also check the prob size, and if it exists.
        if probability is not None:
            assert probability.shape[0] <= self._probability_size, "Size of prob vector is greater than the default."

        # NOTE @kv: Do we have a better round-off?
        assert self._current < self._container_size, "The embedding container is out of capacity!"

        if not isinstance(embedding, (np.ndarray, np.generic)):
            raise TypeError('Legal dtype of embedding is numpy array.')

        self._embeddings[self._current, ...] = embedding

        if probability is not None:
            self._probabilities[self._current, ...] = probability

        if attributes is not None:
            self._attribute_by_instance[instance_id] = attributes
            if not isinstance(attributes, dict):
                print('WARNING: given attributes must be a dictionary.')
            self._attribute_table.add(instance_id, attributes)
            for attr_name, attr_value in attributes.items():
                if attr_value not in self._instance_by_attribute:
                    self._instance_by_attribute[attr_value] = []
                self._instance_by_attribute[attr_value].append(instance_id)

        self._index_by_instance_id[instance_id] = self._current
        self._label_by_instance_id[instance_id] = label_id
        self._label_name_by_instance_id[instance_id] = label_name
        self._instance_id_by_label[label_id].append(instance_id)
        self._instance_ids.append(instance_id)
        self._label_ids.append(label_id)
        self._label_names.append(label_name)
        self._filename_strings.append(filename)
        self._current += 1

    def createIndex(self, recreate=False):
        """Create pandas.DataFrame index table from index buffer
        """
        if self.has_index and recreate:
            print('NOTICE: internal pandas DataFrame is created already')
            return
        internals = self._fetch_internals()
        self._index_df = pd.DataFrame(internals)
        attr_df = self._attribute_table.DataFrame
        if not attr_df.empty:
            self._index_df = pd.merge(self._index_df, attr_df,
                                      on=container_fields.instance_ids,
                                      how='inner')

    def get_embedding_by_instance_ids(self, instance_ids):
        """Fetch batch of embedding vectors by given instance ids."""
        if not (type(instance_ids) is int or type(instance_ids) is list):
            if isinstance(instance_ids, (np.ndarray, np.generic)):
                instance_ids = instance_ids.tolist()
            else:
                raise ValueError('instance_ids should be int or list.')
        if isinstance(instance_ids, int):
            instance_ids = [instance_ids]
        indices = [self._index_by_instance_id[img_id] for img_id in instance_ids]
        return self._embeddings[indices, ...]

    def get_embedding_by_label_ids(self, label_ids):
        """Fetch batch of embedding vectors by given label ids."""
        if not (type(label_ids) is int or type(label_ids) is list):
            raise ValueError('instance_ids should be int or list.')
            if isinstance(label_ids, (np.ndarray, np.generic)):
                label_ids = label_ids.tolist()
            else:
                raise ValueError('instance_ids should be int or list.')
        if isinstance(label_ids, int):
            label_ids = [label_ids]

        indices = []
        for label_id in label_ids:
            for inst_id in self.get_instance_ids_by_label(label_id):
                indices.append(self._index_by_instance_id[inst_id])
        return self._embeddings[indices, ...]

    def get_probability_by_instance_ids(self, instance_ids):
        """Fetch batch of prob vectors by given instance ids."""
        if self._probability_size == 0:
            return np.asarray([])
        if not (type(instance_ids) is int or type(instance_ids) is list):
            if isinstance(instance_ids, (np.ndarray, np.generic)):
                instance_ids = instance_ids.tolist()
            else:
                raise ValueError('instance_ids should be int or list.')
        if isinstance(instance_ids, int):
            instance_ids = [instance_ids]
        indices = [self._index_by_instance_id[img_id] for img_id in instance_ids]
        return self._probabilities[indices, ...]

    def get_probability_by_label_ids(self, label_ids):
        """Fetch batch of prob vectors by given label ids."""
        if self._probability_size == 0:
            return np.asarray([])
        if not (type(label_ids) is int or type(label_ids) is list):
            raise ValueError('instance_ids should be int or list.')
            if isinstance(label_ids, (np.ndarray, np.generic)):
                label_ids = label_ids.tolist()
            else:
                raise ValueError('instance_ids should be int or list.')
        if isinstance(label_ids, int):
            label_ids = [label_ids]
        indices = []
        for label_id in label_ids:
            for inst_id in self.get_instance_ids_by_label(label_id):
                indices.append(self._index_by_instance_id[inst_id])
        return self._probabilities[indices, ...]

    # NOTE: change name?
    def get_label_by_instance_ids(self, instance_ids):
        """Fetch the label_ids from given instance_ids."""
        if isinstance(instance_ids, list):
            return [self._label_by_instance_id[img_id] for img_id in instance_ids]
        elif isinstance(instance_ids, int):
            return self._label_by_instance_id[instance_ids]
        elif isinstance(instance_ids, (np.ndarray, np.generic)):
            return [self._label_by_instance_id[img_id] for img_id in instance_ids.tolist()]
        else:
            raise TypeError('instance_ids should be int, list or array.')

    def get_label_name_by_instance_ids(self, instance_ids):
        """Fetch the label_names from given instance_ids."""
        if isinstance(instance_ids, list):
            return [self._label_name_by_instance_id[img_id] for img_id in instance_ids]
        elif isinstance(instance_ids, int):
            return self._label_name_by_instance_id[instance_ids]
        elif isinstance(instance_ids, (np.ndarray, np.generic)):
            return [self._label_name_by_instance_id[img_id] for img_id in instance_ids.tolist()]
        else:
            raise TypeError('instance_ids should be int, list or array.')

    def get_instance_ids_by_label(self, label_id):
        """Fetch the instance_ids from given label_id."""
        if not np.issubdtype(type(label_id), np.integer):
            raise ValueError('Query label id should be integer.')
        return self._instance_id_by_label[label_id]

    def get_instance_ids_by_exclusive_label(self, label_id):
        """Fetch instance_ids except given label_id."""
        if not np.issubdtype(type(label_id), np.integer):
            raise ValueError('Query label id should be integer.')
        exclusive_label_ids = [_id for _id in set(self._label_ids) if _id != label_id]
        return self.get_instance_ids_by_label_ids(exclusive_label_ids)

    def get_instance_ids_by_label_ids(self, label_ids):
        """Fetch the instance_ids from given label_id."""
        if not (type(label_ids) is int or type(label_ids) is list):
            raise ValueError('instance_ids should be int or list.')
        if isinstance(label_ids, int):
            label_ids = [label_ids]
        _instance_ids = []
        for label_id in label_ids:
            _instance_ids.extend(self._instance_id_by_label[label_id])
        return _instance_ids

    def get_filename_strings_by_instance_ids(self, instance_ids):
        """Fetch the label_names from given instance_ids."""
        if isinstance(instance_ids, list):
            return [self._filename_strings[self._index_by_instance_id[inst_id]]
                    for inst_id in instance_ids]
        if isinstance(instance_ids, int):
            return self._filename_strings[self._index_by_instance_id[instance_ids]]
        if isinstance(instance_ids, (np.ndarray, np.generic)):
            return [self._filename_strings[self._index_by_instance_id[inst_id]]
                    for inst_id in instance_ids.tolist()]
        raise TypeError('instance_ids should be int, list or array.')

    @property
    def embeddings(self):
        return self._embeddings[:self._current]

    @property
    def probabilities(self):
        # get logits up to current index
        if self._probabilities is not None:
            return self._probabilities[:self._current]
        return np.array([])

    @property
    def instance_ids(self):
        # get all instance_ids in container
        return self._instance_ids

    @property
    def attributes(self):
        return list(self._instance_by_attribute.keys())

    @property
    def label_ids(self):
        return self._label_ids

    @property
    def label_id_set(self):
        return list(set(self.label_ids))

    @property
    def label_names(self):
        return self._label_names

    @property
    def filename_strings(self):
        return self._filename_strings

    @property
    def label_name_set(self):
        return list(set(self.label_names))

    @property
    def labelmap(self):
        # id to name
        if self.label_names and self.label_ids:
            labelmap = {}
            for _name, _id in zip(self.label_names, self.label_ids):
                if _name not in labelmap:
                    labelmap[_id] = _name
                else:
                    if labelmap[_id] != _name:
                        # or just print
                        raise ValueError('label name:{} (!={}) is not consistent for id:{}!'.format(
                            _name, labelmap[_name], _id))
            return labelmap
        return {}

    @property
    def has_index(self):
        # Consider both attribute table and index df
        if self._index_df is None:
            return False
        return True

    @property
    def DataFrame(self):
        if not self.has_index:
            self.createIndex()
        return self._index_df

    @property
    def meta_dict(self):
        """Fetch meta_dict used for Cradle.
           Save all dataframe content into meta_dict.
        """
        _meta_dict = {}
        for k, v in self.DataFrame.to_dict('list').items():
            _meta_dict[k] = np.vstack(v)
        return _meta_dict

    @property
    def instance_id_groups(self):
        return self._instance_id_by_label

    @property
    def index_by_instance_ids(self):
        return self._index_by_instance_id

    @property
    def embedding_size(self):
        return self._embedding_size

    @property
    def probability_size(self):
        return self._probability_size

    @property
    def counts(self):
        return self._current

    def clear(self):
        # clear dictionaries
        self._init_internals()
        self._init_arrays(self._container_size,
                          self._embedding_size,
                          self._probability_size)
        self._embeddings = None
        self._probabilities = None
        self._index_df = None
        self._attribute_table.clear()
        print('Clear {}'.format(self._name))

    def save(self, path):
        """Save embedding to disk"""
        # Save as feature_object
        feature_exporter = FeatureObject()
        if self.instance_ids:
            feature_exporter.instance_ids = np.asarray(self.instance_ids)

        if self.filename_strings:
            feature_exporter.filename_strings = np.asarray(self.filename_strings)

        if self.label_ids:
            feature_exporter.label_ids = np.asarray(self.label_ids)

        if self.label_names:
            feature_exporter.label_names = np.asarray(self.label_names)

        if self.counts > 0:
            feature_exporter.embeddings = self.embeddings
            print('Export embedding with shape: {}'.format(self.embeddings.shape))

        if self.counts > 0 and self._probabilities is not None:
            feature_exporter.probabilities = self.probabilities
            print('Export probabilities with shape: {}'.format(self.probabilities.shape))

        feature_exporter.save(path)
        print("Save all extracted features at \'{}\'".format(path))

        if not self.has_index:
            self.createIndex()

        # Save attributes
        attr_table_path = os.path.join(path, 'attribute_table.csv')
        self._attribute_table.save(attr_table_path)
        print("Save all attributes into \'{}\'".format(attr_table_path))

        # Save details
        detail_table_path = os.path.join(path, 'indexes.csv')
        self._index_df.to_csv(detail_table_path)
        print("Save detailed indexed into \'{}\'".format(detail_table_path))

    def load(self, path):
        """Load embedding from disk"""
        # Create FeatureObject
        feature_importer = FeatureObject()
        feature_importer.load(path)

        # type check
        assert feature_importer.label_ids is None or feature_importer.label_ids.size > 0, 'label_ids cannot be empty'
        assert feature_importer.embeddings is None or feature_importer.embeddings.size > 0, 'embeddings cannot be empty'
        assert len(feature_importer.embeddings) == len(feature_importer.label_ids)

        container_size = n_instances = len(feature_importer.label_ids)
        embedding_size = feature_importer.embeddings.shape[1]
        probability_size = feature_importer.probabilities.shape[1] if feature_importer.probabilities is not None else 0

        self._init_internals()
        self._init_arrays(container_size,
                          embedding_size,
                          probability_size)

        # Give sequential instance_ids if not specified
        if feature_importer.instance_ids is None or feature_importer.instance_ids.size == 0:
            instance_ids = np.arange(n_instances)
        else:
            assert len(feature_importer.instance_ids) == n_instances
            instance_ids = feature_importer.instance_ids

        label_names = feature_importer.label_names
        filename_strings = feature_importer.filename_strings
        probabilities = feature_importer.probabilities

        # Create AttributeTable
        csv_file_path = os.path.join(path, 'attribute_table.csv')
        csv_reader = None
        if not os.path.exists(csv_file_path):
            print('NOTICE: {} contains no attribute table'.format(csv_file_path))
        else:
            csv_reader = CsvReader({'path': csv_file_path})

        for idx, instance_id in enumerate(instance_ids):
            label_id = feature_importer.label_ids[idx]
            embedding = feature_importer.embeddings[idx]

            label_name = None
            if label_names is not None:
                label_name = label_names[idx]

            filename = None
            if filename_strings is not None:
                filename = filename_strings[idx]

            probability = None
            if probabilities is not None:
                probability = probabilities[idx]

            attributes = None
            if csv_reader is not None:
                attributes = csv_reader.query_attributes_by_instance_id(instance_id)

            self.add(instance_id=instance_id,
                     label_id=label_id,
                     label_name=label_name,
                     embedding=embedding,
                     probability=probability,
                     attributes=attributes,
                     filename=filename)
        print('Container initialized.')

    def load_pkl(self, pkl_path):
        """Load embeddings & internals from cradle EmbeddingDB format
          NOTE: This function would not verify md5 hash code.
          Args:
            pkl_path: A string, path to the pickle file
        """
        with open(pkl_path, 'rb') as pkl_file:
            db_data = pickle.load(pkl_file)

        embeddings = None
        if container_fields.embeddings not in db_data:
            raise KeyError('missing {required} in {db_keys}'.format(
                required=container_fields.embeddings,
                db_keys=list(db_data.keys())))
        embeddings = db_data[container_fields.embeddings]
        if not isinstance(embeddings, (np.ndarray, np.generic)):
            embeddings = np.asarray(embeddings, np.float32)
        assert len(embeddings.shape) == 2, 'Embeddings should be 2D np array'
        container_size, embedding_size = embeddings.shape
        probability_size = 0
        self._init_internals()
        self._init_arrays(container_size,
                          embedding_size,
                          probability_size)

        meta_dict = None
        if container_fields.meta in db_data:
            meta_dict = db_data[container_fields.meta]
        assert meta_dict is not None, '{} contains no meta data'.format(pkl_path)
        assert container_fields.label_ids in meta_dict, 'Label ids cannot found in meta_dict'

        extra_keys = [k for k in meta_dict.keys()
                      if k not in [container_fields.label_names,
                                   container_fields.instance_ids,
                                   container_fields.label_ids,
                                   container_fields.filename_strings]]
        if container_fields.instance_ids in meta_dict:
            instance_ids = np.squeeze(meta_dict[container_fields.instance_ids])
        else:
            instance_ids = np.arange(container_size)
        label_ids = meta_dict[container_fields.label_ids]
        filename_strings, label_names = None, None
        if container_fields.label_names in meta_dict:
            label_names = meta_dict[container_fields.label_names]
        if container_fields.filename_strings in meta_dict:
            filename_strings = meta_dict[container_fields.filename_strings]

        for idx, instance_id in enumerate(instance_ids):
            label_id = label_ids[idx]
            embedding = embeddings[idx]

            label_name = None
            if label_names is not None:
                label_name = label_names[idx]

            filename = None
            if filename_strings is not None:
                filename = filename_strings[idx]

            probability = None
            attributes = {}
            for k in extra_keys:
                attributes[k] = meta_dict[k][idx][0]
            self.add(instance_id=instance_id,
                     label_id=label_id,
                     label_name=label_name,
                     embedding=embedding,
                     probability=probability,
                     attributes=attributes,
                     filename=filename)
        print('Container initialized.')

    # Add new functions
    def get_instance_id_by_attribute(self, attribute_name):
        """
          Args:
            attribute_name: string
          Return:
            instance_ids: list, empty if query can not be found
        """
        if attribute_name in self._instance_by_attribute:
            return self._instance_by_attribute[attribute_name]
        return []

    def get_attribute_by_instance_id(self, instance_id):
        """
          Args:
            instance_id: int
          Return:
            attributes: list, empty if query can not be found
        """
        if instance_id in self._attribute_by_instance:
            return self._attribute_by_instance[instance_id]
        return []

    def get_instance_id_by_group_command(self, command):
        """
          Args:
            command: string of query command in defined format
                command = 'A+B-C'
                where A, B, C are attribute_name
          Return:
            results: list of integer
          NOtE: Special commands:
            - all
            (TODO)- all_class
            (TODO)- all_attribute
        """
        # Special Cases
        if command == attr_field.All:
            return self.instance_ids

        # General Case
        executable_codes = self._translate_command_to_executable(command)
        self._interpreter.run_code(executable_codes)
        results = self._interpreter.fetch()
        self._interpreter.clear()
        return results

    def get_instance_id_by_cross_reference_command(self, command):
        """Parse one more line than group command
          Args:
            command: string of query command in defined format
                command = '(A+B)->C+D' where A, B, C, D are attribute_name
          Returns:
            source: list of integer
            target: list of integer
        """
        def _split_cross_reference_command(command):
            m = re.match(r'(.+)->(.+)', command)
            source = m.group(1)
            target = m.group(2)
            return source, target

        source_command, target_command = _split_cross_reference_command(command)
        source_result = self.get_instance_id_by_group_command(source_command)
        target_result = self.get_instance_id_by_group_command(target_command)
        return source_result, target_result

    def _translate_command_to_executable(self, single_line_command):
        executable_command = {
            interpreter_field.instructions: [],
            interpreter_field.values: [],
            interpreter_field.names: [],
        }

        def _translate_command(operation):
            """Two operators are legal: +, -"""
            operation = operation.replace(' ', '')
            operation = re.sub(r'[(){}]', '', operation)
            op_list = re.split(r'\w', operation)
            operands = re.split(r'\+|\-', operation)
            op_list = [op for op in op_list if op in ['+', '-']]
            return operands, op_list

        def _put_variable_in_stack(name, a_list):
            nonlocal stack_pointer
            executable_command[interpreter_field.instructions].append(
                (interpreter_field.LOAD_LIST, stack_pointer))
            executable_command[interpreter_field.instructions].append(
                (interpreter_field.STORE_NAME, stack_pointer))
            executable_command[interpreter_field.instructions].append(
                (interpreter_field.LOAD_NAME, stack_pointer))
            executable_command[interpreter_field.names].append(name)
            executable_command[interpreter_field.values].append(a_list)
            stack_pointer += 1

        def _put_command_in_stack(operator):
            executable_command[interpreter_field.instructions].append(
                (operator, None))

        stack_pointer = 0
        operand_names, op_list = _translate_command(single_line_command)
        # push first variable

        attr_name = operand_names.pop()
        instance_ids = self.get_instance_id_by_attribute(attr_name)
        _put_variable_in_stack(attr_name, instance_ids)

        if len(op_list) == len(operand_names):
            for attr_name, op_symbol in zip(operand_names, op_list):
                ###
                instance_ids = self.get_instance_id_by_attribute(attr_name)
                _put_variable_in_stack(attr_name, instance_ids)
                instruction = InstructionSymbolTable[op_symbol]
                _put_command_in_stack(instruction)
        return executable_command
