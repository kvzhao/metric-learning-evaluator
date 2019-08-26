"""Interpreter for General list operations

    Supported operations:
        +: Join
        -: Remove
        &: And
    TODO:
        ~: Exclusive
    TODO: @kv redesign is needed. Reconsider the role when pandas is used.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import re

from metric_learning_evaluator.core.standard_fields import InterpreterStandardField as interpreter_field

# Change name
InstructionSymbolTable = {
    '+': 'JOIN',
    '&': 'AND',
    '-': 'REMOVE',
    '#': 'PRINT',
}


def _split_command_by_operator(operation):
    """
      Split operations and operands
        source.Vis + source.uS85 & type.seen -> 
        ['source.Vis', 'source.uS85', 'type.seen'], ['+', '&']
    """
    operation = operation.replace(' ', '')
    operation = re.sub(r'[(){}]', '', operation)
    op_list = re.split(r'\w', operation)
    operands = re.split(r'\+|\-|\&', operation)
    op_list = [op for op in op_list if op in ['+', '-', '&']]
    return operands, op_list


class CommandExecutor(object):
    """
      Is it a Pandas Wrapper?
    """
    def __init__(self, dataframe):
        self._df = dataframe

        # operate data given standard commands
        self._interpreter = Interpreter()

    def execute(self, command):
        """
          Args:
            command: string
                One line of commands - group or cross reference
          Returns:
            dict of list?
            list of list?
            name, list?
          NOTE
            push values in `what_to_execute` dict
        """
        pass


def command_to_executable_codes(single_line_command):
    """
      single_line_command format
        - operands
    """
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
    instance_ids = self.get_instance_id_by_attribute_value(attr_name)
    _put_variable_in_stack(attr_name, instance_ids)

    if len(op_list) == len(operand_names):
        for attr_name, op_symbol in zip(operand_names, op_list):
            ###
            instance_ids = self.get_instance_id_by_attribute_value(attr_name)
            _put_variable_in_stack(attr_name, instance_ids)
            instruction = InstructionSymbolTable[op_symbol]
            _put_command_in_stack(instruction)
    return executable_command


class Interpreter(object):
    """
      Operate data with given standard operation commands.
      e.g
      execution_commands = {
            'instructions': [
                ('LOAD_LIST', 0),
                ('STORE_NAME', 0),
                ('LOAD_LIST', 1),
                ('STORE_NAME', 1),
                ('LOAD_NAME', 0),
                ('LOAD_NAME', 1),
                ('JOIN', None),
            ],
            'values': [
                [1, 3, 5],
                [2, 3, 5, 7],
            ],
            'names': [
                'A',
                'B',
                'C',
            ],
        }
      }
    """
    def __init__(self):
        self.stack = []
        self.environment = {}

    def STORE_NAME(self, name):
        a_list = self.stack.pop()
        self.environment[name] = a_list

    def LOAD_NAME(self, name):
        a_list = self.environment[name]
        self.stack.append(a_list)

    def LOAD_LIST(self, a_list):
        if not isinstance(a_list, list):
            return
        self.stack.append(a_list)

    def JOIN(self):
        # join two list
        arr_1 = self.stack.pop()
        arr_2 = self.stack.pop()
        result = arr_1 + list(set(arr_2) - set(arr_1))
        self.stack.append(result)

    def AND(self):
        # logic and two list
        arr_1 = self.stack.pop()
        arr_2 = self.stack.pop()
        result = list(set(arr_1) & set(arr_2))
        self.stack.append(result)

    def REMOVE(self):
        # mutual exclusive two list
        arr_1 = self.stack.pop()
        arr_2 = self.stack.pop()
        # TODO @kv: checkout this logic
        result = list(set(arr_1) ^ set(arr_2))
        self.stack.append(result)

    def PRINT(self):
        ans = self.stack.pop()
        print(ans)
        self.stack.append(ans)

    def fetch(self):
        if self.stack:
            return self.stack.pop()
        return []

    def clear(self):
        self.stack = []
        self.environment = {}

    def parse_argument(self, instruction, argument, what_to_execute):
        # return a dict: `what_to_execute`
        values = ['LOAD_LIST']
        names = ['LOAD_NAME', 'STORE_NAME']
        if instruction in values:
            argument = what_to_execute['values'][argument]
        elif instruction in names:
            argument = what_to_execute['names'][argument]
        return argument

    def run_code(self, what_to_execute):
        instructions = what_to_execute['instructions']
        for each_step in instructions:
            instruction, argument = each_step
            argument = self.parse_argument(instruction, argument, what_to_execute)
            bytecode_method = getattr(self, instruction)
            if argument is None:
                bytecode_method()
            else:
                bytecode_method(argument)
