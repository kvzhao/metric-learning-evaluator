"""Interpreter for General list operations

    Supported operations:
        +: Join
        -: Remove
        &: And
    TODO:
        ~: Exclusive
"""

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import re

class InterpreterStandardField:
    LOAD_LIST = 'LOAD_LIST'
    STORE_NAME = 'STORE_NAME'
    LOAD_NAME = 'LOAD_NAME'
    JOIN = 'JOIN'
    REMOVE = 'REMOVE'
    PRINT = 'PRINT'
    instructions = 'instructions'
    values = 'values'
    names = 'names'

field = InterpreterStandardField

InstructionSymbolTable = {
    '+': 'JOIN',
    '&': 'AND',
    '-': 'REMOVE',
    '#': 'PRINT',
}

class Interpreter(object):

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
        if (not isinstance(a_list, list)):
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
        result = list(set(arr_1)^set(arr_2))
        self.stack.append(result)

    def PRINT(self):
        ans = self.stack.pop()
        print(ans)
        self.stack.append(ans)

    def fetch(self):
        if self.stack:
            return self.stack.pop()
        else:
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