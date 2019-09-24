

import unittest

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from metric_learning_evaluator.utils.interpreter import Interpreter


class InterpreterTestCase(unittest.TestCase):

    def test_join(self):
        what_to_execute = {
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

        interpreter = Interpreter()
        interpreter.run_code(what_to_execute)
        outcome = interpreter.fetch()
        target = [1, 2, 3, 5, 7]

        self.assertCountEqual(outcome, target)

if __name__ == "__main__":
    unittest.main()
