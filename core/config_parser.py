import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import yaml
import inspect

from core.eval_standard_fields import ConfigStandardFields as config_fields
from core.eval_standard_fields import EvaluationStandardFields as eval_fields
from core.eval_standard_fields import AttributeStandardFields as attr_fields
from core.eval_standard_fields import MetricStandardFields as metric_fields


class ConfigParser(object):
    """
      User Scenario:
        1. load from given config path
        2. get useful values to set containers
        3. get hyper-parameters for evaluation functions
        
        NOTE:
          evaluation_type: classification vs. evaluation_name: ClassificationEvaluation
    """

    def __init__(self, config_path):
        """
          Args:
            config_path, string: Path to the yaml file.
        """

        try:
            with open(config_path, 'r') as fp:
                self._configs = yaml.load(fp)
        except:
            raise IOError('Can not load yaml from {}.'.format(config_path))
            # TODO: create default config instead of error.

        minimum_requirements_of_config = [
            config_fields.container_size,
            config_fields.embedding_size,
            config_fields.logit_size,
            config_fields.evaluation,
            config_fields.database,
        ]
        for req in minimum_requirements_of_config:
            if not req in self._configs:
                raise ValueError('''The given configuration is not legal. It should
                    contain `evaluation`, `container_size`, `embedding_size`, `logit_size` and `database`.''')
        if not self._configs[config_fields.evaluation]:
            raise ValueError('No any evaluation is given in eval_config.')

        if not self._configs[config_fields.database]:
            print ('WARNING: No database is assigned, some evaluations can not be executed.')
            # NOTE @kv: Should we ignore given attribute names & use category as attr_name?
        
        # Make sure self._config is legal dict before parsing.
        self._parse()


    def _parse(self):

        # Fetch valid evaluation names
        self._evaluation_names = self._configs[config_fields.evaluation]

        # Save evaluator system information
        self._embedding_size = self._configs[config_fields.embedding_size]
        self._container_size = self._configs[config_fields.container_size]
        self._logit_size = self._configs[config_fields.logit_size]
        self._database = self._configs[config_fields.database]

        # Collect attributes
        required_attributes = []
        for eval_type in self._evaluation_names:
            per_eval_config = self._configs[config_fields.evaluation][eval_type]
            if not per_eval_config:
                continue
            if attr_fields.attr_key in per_eval_config:
                attrs = per_eval_config[attr_fields.attr_key]
                required_attributes.extend(attrs)
        self._required_attributes = list(set(required_attributes))

        # Extract all defined names in metric standard fields
        metric_items = inspect.getmembers(metric_fields, lambda a: not(inspect.isroutine(a)))
        metric_items = [a for a in metric_items if not(a[0].startswith('__') and a[0].endswith('__'))]
        self._metric_items = [v for _, v in metric_items]

    @property
    def database_type(self):
        return self._database

    @property
    def embedding_size(self):
        return self._embedding_size

    @property
    def container_size(self):
        return self._container_size

    @property
    def logit_size(self):
        return self._logit_size

    @property
    def evaluation_names(self):
        return self._evaluation_names

    @property
    def required_attributes(self):
        return self._required_attributes

    def get_per_eval_config(self, eval_name):
        """
          Args:
            evaluation_name, string: Name of the evaluation object.
                e.g. MockEvaluation, same as the class.
          Return:
            per_eval_config, dict:
                Description of the per evaluation.
        """
        if eval_name in self.evaluation_names:
            return self._configs[config_fields.evaluation][eval_name] 
        else:
            # TODO: What should we do here?
            return None

    def has_attribute(self, eval_name):
        _attrs = self.get_per_eval_attributes(eval_name)

        if not _attrs or attr_fields.all_classes in _attrs:
            return False
        else:
            return True

    def get_per_eval_attributes(self, eval_name):
        """
          Args:
            eval_name, string: Name of the evaluation object.
          Return:
            List of attributes to be evaluated at given evaluation.
        """
        per_eval = self.get_per_eval_config(eval_name)
        if not per_eval:
            return []
        if attr_fields.attr_key in per_eval:
            return per_eval[attr_fields.attr_key]
        else:
            return [attr_fields.all_classes]

    def get_per_eval_metrics(self, eval_name):
        """
          Args:
            eval_name, string: Name of the evaluation object.
          Return:
            Dict of list of metrics and corresponding thresholds.
        """
        per_eval = self.get_per_eval_config(eval_name)
        if not per_eval:
            return {}
        # return the dict which key in metric standard fields
        metrics_dict = {}
        for k, v in per_eval.items():
            if k in self._metric_items:
                metrics_dict[k] = v
        return metrics_dict