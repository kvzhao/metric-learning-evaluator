import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import json

# TODO @dennis.liu :
#   Should I use the base wrapper?
class JsonWrapper(object):
    def __init__(self, db_config):
        path = db_config.get("path",None)
        # grouping_rules_file_path = db_config.get("grouping_rules", None)

        if path is None :
            raise NameError("path is required")
        # if grouping_rules_file_path is None :
        #     raise NameError("grouping_rules_file_path is required")

        if os.path.exists(path):
            with open(path) as f:
                self._attributes_table = json.load(f)
            print("{} is loaded".format(path))
        else:
            raise FileNotFoundError("DB path: {} not found".format(path))

        # if os.path.exists(grouping_rules_file_path):
        #     with open(grouping_rules_file_path) as f:
        #         self._grouping_rules = json.load(f)
        #     print("{} is loaded".format(grouping_rules_file_path))
        # else:
        #     raise FileNotFoundError("DB path: {} not found".format(grouping_rules_file_path))

    def query_attributes_by_instance_id(self, instance_id):

        attributes = self._attributes_table.get(str(instance_id),[])
        return attributes
