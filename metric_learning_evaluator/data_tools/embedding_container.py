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


from metric_learning_evaluator.utils.switcher import switch
from metric_learning_evaluator.query.general_database import QueryInterface
from metric_learning_evaluator.utils.interpreter import Interpreter
from metric_learning_evaluator.utils.interpreter import InstructionSymbolTable
from metric_learning_evaluator.utils.interpreter import InterpreterStandardField as interpreter_field
from metric_learning_evaluator.query.standard_fields import AttributeStandardFields as attr_field


class EmbeddingContainer(object):
    """The Data Container for Embeddings & Logit (instance_id, label_id, embedding vector).

      operations:
        - add: put one datum in the container
        - embeddings: get all embedding vectors exist in the container
        - get_embedding_by_instance_ids: query embeddings by instance_ids
        - get_label_by_instance_ids: query labels by instance_ids
        - clear: clear the internal buffer
    
      NOTE: We CAN NOT confirm the orderness of logits & embedding consistent with instance_ids.
      TODO @kv: implement save & load for data container.
      TODO @kv: Error-handling when current exceeds container_size
      TODO @kv: instance_id can be `int` or `filename`, this is ambiguous
      TODO @kv: maybe we should add filename in container.
      TODO @kv: update or init container with blob of numpy array

    """
    def __init__(self, embedding_size, prob_size,
                 container_size=10000, name='embedding_container'):
        """Constructor of the Container.

          Args:
            embedding_size, int:
                Dimension of the embedding vector, e.g. 1024 or 2048.
            prob_size, int:
                Disable this by giving size equals to 0.
            probabilities: an ndarray of probabilities each class, disable this by giving size equals to 0.
                It prefers passing top_k scores.
            container_size, int:
                Number of embedding vector that container can store.
            name, str:
                The name string is used for version control.
        
        """
        self._embedding_size = embedding_size
        self._prob_size = prob_size
        self._container_size = container_size
        # TODO: Check the dimensionality of size
        self._embeddings = np.empty((container_size, embedding_size), dtype=np.float32)
        if prob_size == 0:
            self._probs = None
        else:
            self._probs = np.empty((container_size, prob_size), dtype=np.float32)

        self._init_internals()

        # used for parsing commands
        self._interpreter = Interpreter()

        self._name = name
        self._current = 0

    def __repr__(self):
        _content = '===== {} =====\n'.format(self._name)
        _content += 'embeddings: {}'.format(self._embeddings.shape)

    def _init_internals(self):
        # maps index used in numpy array and instance_id list
        self._label_by_instance_id = {}
        self._index_by_instance_id = {}
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

    def add(self, instance_id, label_id, embedding,
            prob=None, attributes=None, label_name=None, filename=None):
        """Add instance_id, label_id and embeddings.
        TODO: Add attributes, label_name, filename and more.
          Args:
            instance_id: int
                Unique instance_id which can not be repeated in the container.
            label_id: int
                Index of given class corresponds to the instance.
            embedding: 1D numpy array:
                One dimensional embedding vector with size less than self._embedding_size.
            prob: 1D numpy array:
                One dimensional vector which records class-wise scores.
            attributes: List of strings
                List of attributes corresponding to the given instance_id
            label_name: String
                Human-realizable content of given label_id
            filename: String
                The filename or filepath to the given instance_id.
        """
        # check type of label_id, instance_id, TODO: Use more elegant way
        try:
            label_id = int(label_id)
            instance_id = int(instance_id)
        except:
            raise TypeError("The label id or instance id has wrong type")

        # assertions: embedding size, 
        assert embedding.shape[0] <= self._embedding_size, "Size of embedding vector is greater than the default."
        # TODO @kv: Also check the prob size, and if it exists.
        if prob is not None:
            assert prob.shape[0] <= self._prob_size, "Size of prob vector is greater than the default."

        # NOTE @kv: Do we have a better round-off?
        assert self._current < self._container_size, "The embedding container is out of capacity!"

        if not isinstance(embedding, (np.ndarray, np.generic)):
            raise TypeError ('Legal dtype of embedding is numpy array.')

        self._embeddings[self._current, ...] = embedding

        if prob is not None:
            self._probs[self._current, ...] = prob

        if attributes is not None:
            if isinstance(attributes, str):
                attributes = [attributes]
            if not all(isinstance(_attr, str) for _attr in attributes):
                raise ValueError('attributes type should be str or list of str.')
            #TODO: add one more attributes `all` into the container! (also works for attributes is None case)
            self._attribute_by_instance[instance_id] = attributes
            for _attr in attributes:
                if _attr in self._instance_by_attribute:
                    self._instance_by_attribute[_attr].append(instance_id)
                else:
                    self._instance_by_attribute[_attr] = [instance_id]

        # NOTE: same instance_id maps to many embedding!?
        self._index_by_instance_id[instance_id] = self._current
        self._label_by_instance_id[instance_id] = label_id
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
        if self._prob_size == 0:
            return np.asarray([])
        if not (type(instance_ids) is int or type(instance_ids) is list):
            if isinstance(instance_ids, (np.ndarray, np.generic)):
                instance_ids = instance_ids.tolist()
            else:
                raise ValueError('instance_ids should be int or list.')
        if isinstance(instance_ids, int):
            instance_ids = [instance_ids]
        indices = [self._index_by_instance_id[img_id] for img_id in instance_ids]
        return self._probs[indices, ...]

    def get_probability_by_label_ids(self, label_ids):
        """Fetch batch of prob vectors by given label ids."""
        if self._prob_size == 0:
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
        return self._probs[indices, ...]

    def get_label_by_instance_ids(self, instance_ids):
        """Fetch the labels from given instance_ids."""
        if isinstance(instance_ids, list):
            return [self._label_by_instance_id[img_id] for img_id in instance_ids]
        elif isinstance(instance_ids, int):
            return self._label_by_instance_id[instance_ids]
        elif isinstance(instance_ids, (np.ndarray, np.generic)):
            return [self._label_by_instance_id[img_id] for img_id in instance_ids.tolist()]
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

    @property
    def embeddings(self):
        # get embeddings up to current index
        return self._embeddings[:self._current]

    @property
    def probs(self):
        # get logits up to current index
        return self._probs[:self._current]

    @property
    def instance_ids(self):
        # get all instance_ids in container
        return self._instance_ids
    @property
    def label_ids(self):
        return self._label_ids

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
    def prob_size(self):
        return self._prob_size

    @property
    def counts(self):
        return self._current

    def clear(self):
        # clear dictionaries
        self._init_internals()
        # reset the current index, rewrite array instead of clear elements.
        self._current = 0
        print ('Clear embedding container.')

    def save(self, path):
        # Save as feature_object
        pass

    def load(self, path):
        pass

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
        else:
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