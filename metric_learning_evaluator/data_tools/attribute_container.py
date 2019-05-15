"""
    Define data containers for the metric learning evaluator.

    Brief intro:

        AttributeContainer:
            Data object for maintaining attribute table in each EvaluationObject.

    @bird, dennis, kv
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from abc import ABCMeta
from abc import abstractmethod
import collections
from collections import defaultdict

import numpy as np
import re

from metric_learning_evaluator.query.general_database import QueryInterface

from metric_learning_evaluator.utils.switcher import switch

from metric_learning_evaluator.utils.interpreter import Interpreter
from metric_learning_evaluator.utils.interpreter import InstructionSymbolTable
from metric_learning_evaluator.utils.interpreter import InterpreterStandardField as interpreter_field
from metric_learning_evaluator.query.standard_fields import AttributeStandardFields as attr_field


class AttributeContainer(object):
    """The Data Container for Attributes (domain & property)

      Usage:
        User would add instance_id & corresponding attributes (domain & property) into container
        then, the managed structure can be returned.

        Attribute is devidded into two types:
          - Property:
                Property acts as filtering condition like tag
          - Domain:
                Domain serves as grouping condition in evaluation
        Both types will be `flatten` as attribute_name in the container.

        General query interface will turn structure attributes into tag

        Example:
            instance_id:1
                domain: database
                property:
                    - color: red
                    - shape: square

            instance_id:2
                domain: database
                property:
                    - color: yellow
                    - shape: round

            instance_id:3
                domain: query
                property:
                    - color: blue
                    - shape: square

            * `domain` is used for crossing evaluation: `query` to `database`
            * `property` is used as filter:
                - evaluate only `color`
                - evaluate only `shape`
                - evaluate `shape` == square

      Operations:
        - add: put one datum into container

    """

    def __init__(self):
        """
            Two kinds of attribute structures both kinds of attributes will be flatten
        """
        # attribute-id mapping, shallow key-value pair
        self._instance2attr = defaultdict(list)
        # instance_ids with same tag
        self._groups = {}
        self._interpreter = Interpreter()

    @property
    def attribute_names(self):
        _attr_names = list(sorted(self._groups.keys()))
        return _attr_names

    @property
    def instance_to_attribute(self):
        return self._instance2attr

    @property
    def attribute_to_instance(self):
        return self._groups

    # TODO @kv: modify to instance_id: attribute_names
    def add(self, instance_id, attributes):
        """Add the attribute with corresponding groupping indices.

          Args:
            instance_id, int:
                An instance_id is used for query attributes.

            attributes, list of str:
                Attributes of the given instance_id.

          Logic:
            if attr not exist:
                create new key and assign the given list.
            else:
        """
        # type assertion
        if not type(instance_id) is int:
            raise ValueError('instance_id should be an integer.')
        if isinstance(attributes, str):
            attributes = [attributes]
        
        if not all(isinstance(_attr, str) for _attr in attributes):
            raise ValueError('attributes type should be str or list of str.')
        
        self._instance2attr[instance_id] = attributes

        for _attr in attributes:
            if _attr in self._groups:
                self._groups[_attr].append(instance_id)
            else:
                self._groups[_attr] = [instance_id]

    def get_instance_id_by_attribute(self, attribute_name):
        """
          Args:
            attribute_name: string
          Return:
            instance_ids: list, empty if query can not be found
        """
        if attribute_name in self._groups:
            return self._groups[attribute_name]
        else:
            return []

    def get_instance_id_by_group_command(self, command):
        """
          Args:
            command: string of query command in defined format
                command = 'A+B-C'
                where A, B, C are attribute_name
          Return:
            attribute_group:
                dict = {
                    attr_name: [instance_ids]
                }
        """
        executable_codes = self._translate_command_to_executable(command)
        self._interpreter.run_code(executable_codes)
        results = self._interpreter.fetch()
        self._interpreter.clear()
        return {
            command: results
        }

    def get_instance_id_by_cross_reference_command(self, command):
        """Parse one more line than group command
          Args:
            command: string of query command in defined format
                command = '(A+B)->C+D' where A, B, C, D are attribute_name
          Returns:
            source: dict
            target: dict of {attr_name: [instance_ids]}
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

    def clear(self):
        # clear the internal dict.
        self._groups = defaultdict(list)
        self._instance2attr = defaultdict(list)
        print ('Clear attribute container.')