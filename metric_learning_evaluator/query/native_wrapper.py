import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from metric_learning_evaluator.data_tools.attribute_table import AttributeTable


class NativeWrapper(object):
    def __init__(self, database_config):
        path = database_config.get('path', None)

        if path is None :
            raise FileNotFoundError("attribute database path is required")

        # if grouping_rules_file_path is None :
        #     raise NameError("grouping_rules_file_path is required")

        if os.path.exists(path):
            self._database = AttributeTable(path)
        else:
            raise FileNotFoundError("Given database path: {} not found".format(path))

        # if os.path.exists(grouping_rules_file_path):
        #     with open(grouping_rules_file_path) as f:
        #         self._grouping_rules = json.load(f)
        #     print("{} is loaded".format(grouping_rules_file_path))
        # else:
        #     raise FileNotFoundError("DB path: {} not found".format(grouping_rules_file_path))


    def query_attributes_by_instance_id(self, instance_id):
        _attrs = []
        _attrs.extend(self._query_domain_by_instance_id(instance_id))
        _attrs.extend(self._query_property_by_instance_id(instance_id))
        return _attrs

    def _query_domain_by_instance_id(self, instance_id):
        # query tag-like, return a list
        return self._database.query_domain_by_instance_ids(instance_id)

    def _query_property_by_instance_id(self, instance_id):
        # query {attr_name: attr_vale} structure and flatten it to [attr_name.attr_value, ]
        _attr_flatten = []
        attr_dict_list = self._database.query_property_by_instance_ids(instance_id)
        for _attr_dict in attr_dict_list:
            for k, v in _attr_dict.items():
                _attr_flatten.append('.'.join([k, v]))
        return _attr_flatten
