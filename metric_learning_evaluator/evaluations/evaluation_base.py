"""
    The base object and data container for the metric learning evaluator.

    Brief intro:

        EmbeddingContainer: 
            Efficient object which handles the shared (globally) embedding vectors.

        AttributeContainer:
            Data object for maintaining attribute table locally in each EvaluationObejct.

        MetricEvaluationBase:

    @bird, dennis, kv
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from abc import ABCMeta
from abc import abstractmethod

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer

import logging
import numpy as np


class MetricEvaluationBase(object):
    """Interface for Evaluation Object which serves as the functional building block.

        Usage:

    """
    __metaclass__ = ABCMeta

    def __init__(self, eval_config, mode=None):
        """Base Object for Evaluation.
          <Customized>Evaluation is the functional object which 
                      executes computation with metric functions.
          Args:
            config_parse, EvaluationConfigParser:
                Configuration used for the EvaluationObject.
                (which is provided by calling parser.get_eval_config)
        """
        # TODO @kv: remove dash line in variable names
        self.configs = eval_config

        # TODO: Iterator for getting embeddings from given attribute_names
        self._evaluation_name = self.__class__.__name__

        # Fetch all information from eval_config, If not None:
        # preprocessing eval config in each customized evaluation
        self.metrics = self.configs.metric_section
        self.distance_measure = self.configs.distance_measure
        self.attributes = self.configs.attributes
        self.attribute_items = self.configs.attribute_items
        self.sampling = self.configs.sampling_section
        self.option = self.configs.option_section
        # mode: online | offline | None
        self.mode = mode

        self.cross_reference_commands = self.configs.attribute_cross_reference_commands
        self.group_commands = self.configs.attribute_group_commands

    @property
    def evaluation_name(self):
        return self._evaluation_name

    @abstractmethod
    def metric_names(self):
        pass

    def show_configs(self):
        print('{} - Compute {} metrics over attributes: {}'.format(
            self.evaluation_name,
            ', '.join(self.metrics.keys()),
            ', '.join(self.configs.attribute_items)))

    @abstractmethod
    def compute(self, embedding_container):
        """Compute metrics.
          Args:
            embedding_container, EmbeddingContainer:
                The embedding container is necessary.

          Return:
            metrics, dict:
                TODO @kv: Define the standard return format.
        """

        # check the instance type.
        if not isinstance(embedding_container, EmbeddingContainer):
            raise ValueError('Embedded Container is Needed.')

    @abstractmethod
    def save(self, path):
        pass
