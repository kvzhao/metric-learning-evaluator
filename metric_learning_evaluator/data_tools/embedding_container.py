"""
    Define data containers for the metric learning evaluator.

    Brief intro:

        EmbeddingContainer:
            Efficient object which handles the shared (globally) embedding vectors.

        AttributeContainer:
            Data object for maintaining attribute table in each EvaluationObject.

    NOTE: The container is the stack,

    @bird, dennis, kv
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import re
import numpy as np

from abc import ABCMeta
from abc import abstractmethod
import collections
from collections import defaultdict
from metric_learning_evaluator.data_tools.feature_object import FeatureObject
from metric_learning_evaluator.data_tools.attribute_table import AttributeTable

from metric_learning_evaluator.utils.interpreter import Interpreter
from metric_learning_evaluator.utils.interpreter import InstructionSymbolTable
from metric_learning_evaluator.utils.interpreter import InterpreterStandardField as interpreter_field
from metric_learning_evaluator.query.standard_fields import AttributeStandardFields as attr_field


class EmbeddingContainer(object):
    """The Data Container for Embeddings & Probabilities

      Operations:
        - add: put one datum in the container
        - embeddings: get all embedding vectors exist in the container
        - get_embedding_by_instance_ids: query embeddings by instance_ids
        - get_label_by_instance_ids: query labels by instance_ids
        - clear: clear the internal buffer

      Query interfaces:

      = NOTE =======================================================================================
      NOTE: We CAN NOT confirm the orderness of logits & embedding consistent with instance_ids.
      TODO @kv: use pandas dataframe as internals
      TODO @kv: Error-handling when current exceeds container_size
      TODO @kv: instance_id can be `int` or `filename`, this is ambiguous
      TODO @kv: maybe we should add filename in container.
      TODO @kv: update or init container with blob of numpy array
      NOTE @kv: Change the interface this commit!
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
        self._dataframe = None
        self._name = name

        self._init_internals()
        self._init_arrays(self._container_size,
                          self._embedding_size,
                          self._probability_size)

        # used for parsing commands
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
        """Internal Indexes
        """
        # maps index used in numpy array and instance_id list
        self._index_by_instance_id = {}
        self._label_by_instance_id = {}
        self._label_name_by_instance_id = {}
        # orderness should be maintained in _instance_ids
        self._instance_ids = []
        self._label_ids = []
        self._filename_strings = []
        self._label_names = []
        # attribute-id mapping, shallow key-value pair
        self._instance_id_by_label = defaultdict(list)
        self._attribute_by_instance = defaultdict(list)
        # instance_ids with same attribute
        self._instance_by_attribute = {}

    def _init_arrays(self, container_size, embedding_size, probability_size):
        """Internal numpy arrays and array_index"""
        # TODO: Check the dimensionality of size
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
            embedding: 1D numpy array:
                One dimensional embedding vector with size less than self._embedding_size.
            probability: 1D numpy array:
                One dimensional vector which records class-wise scores.
            attributes: List of strings
                List of attributes corresponding to the given instance_id
            label_name: String
                Human-realizable content of given label_id
            filename: String
                The filename or filepath to the given instance_id.
        """
        # check type of label_id, instance_id,
        # TODO: Use more elegant way
        # type check?
        try:
            label_id = int(label_id)
            instance_id = int(instance_id)
        except:
            raise TypeError("The label id or instance id has wrong type")

        # assertions: embedding size, 
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
            if isinstance(attributes, str):
                attributes = [attributes]
            if not all(isinstance(_attr, str) for _attr in attributes):
                raise ValueError('attributes type should be str or list of str.')
            # TODO: add one more attributes `all` into the container! (also works for attributes is None case)
            self._attribute_by_instance[instance_id] = attributes
            for _attr in attributes:
                if _attr in self._instance_by_attribute:
                    self._instance_by_attribute[_attr].append(instance_id)
                else:
                    self._instance_by_attribute[_attr] = [instance_id]

        # NOTE: same instance_id maps to many embedding!?
        self._index_by_instance_id[instance_id] = self._current
        self._label_by_instance_id[instance_id] = label_id
        self._label_name_by_instance_id[instance_id] = label_name
        self._instance_id_by_label[label_id].append(instance_id)
        self._instance_ids.append(instance_id)
        self._label_ids.append(label_id)
        self._label_names.append(label_name)
        self._filename_strings.append(filename)

        self._current += 1

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

    def commit(self):
        """Convert internals to pandas.DataFrame"""
        pass

    @property
    def embeddings(self):
        # get embeddings up to current index
        return self._embeddings[:self._current]

    @property
    def probabilities(self):
        # get logits up to current index
        return self._probabilities[:self._current]

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

        # Save attributes
        if self._attribute_by_instance:
            db_path = os.path.join(path, 'attribute.db')
            attribute_table = AttributeTable(db_path)
            for instance_id, attributes in self._attribute_by_instance.items():
                for attribute in attributes:
                    if '.' in attribute: # attribute
                        name, content = attribute.split('.', 1)
                        attribute_table.insert_property(instance_id, name, content)
                    else: # tag
                        attribute_table.insert_domain(instance_id, attribute)
            attribute_table.commit()
            print("Save all attributes into \'{}\'".format(db_path))

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
        self._init_arrays(container_size, embedding_size, probability_size)

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
        db_path = os.path.join(path, 'attribute.db')
        if not os.path.exists(db_path):
            print('NOTICE: {} contains no attribute table'.format(path))
        attribute_table = AttributeTable(db_path)

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

            properties = attribute_table.query_property_by_instance_ids(int(instance_id))
            domains = attribute_table.query_domain_by_instance_ids(int(instance_id))

            attributes = []
            if properties or domains:
                attributes = ['{}.{}'.format(name, content) for property_ in properties
                              for name, content in property_.items()] + domains
            self.add(instance_id=instance_id,
                     label_id=label_id,
                     label_name=label_name,
                     embedding=embedding,
                     probability=probability,
                     attributes=attributes,
                     filename=filename)

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
        else:
            # seem to be error case
            pass
        return executable_command
