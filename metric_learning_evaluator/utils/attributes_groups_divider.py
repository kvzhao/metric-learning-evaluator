"""
    Used to divide each instances into indexes set for query and db groups from it's attributes(tags)
    @ dennis.liu

"""

import os
import sys

from easydict import EasyDict

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

class AttributesGroupsDividerFields(object):
    grouping_rules = "grouping_rules"
    group_definitions = "group_definitions"
    rank_events = "rank_events"

    comment = "comment"
    filter_attrs = "filter_attrs"
    union_attrs = "union_attrs"

    db_group = "db_group"
    query_group = "query_group"



class AttributesGroupsDivider(object):
    """
    args:
        attributes_container:
            <metric_learning_evaluator.data_tools.attrubute_container>
        grouping_rules <dict>:
            rules of attributes grouping.

    grouping_rules example:
        grouping_rules = [
            # rule1
            {
                field.name: "Fisheyes overall rank",
                field.filter_attrs: "view.pano",
                field.union_attrs: "offset.0,size.0",
                field.query_attrs: "view.fish",
                field.db_attrs: "all"
            },
            # rule2
            ...
        ]
    """

    def __init__(self, attributes_container, grouping_rules=None):
        self._attr_container = attributes_container

        self._fields = AttributesGroupsDividerFields
        assert grouping_rules is not None
        assert grouping_rules.get(self._fields.group_definitions) is not None
        assert grouping_rules.get(self._fields.rank_events) is not None
        self._grouping_rules = grouping_rules

    @property
    def rank_events(self):
        return self._grouping_rules.get(self._fields.rank_events, [])

    @property
    def groups(self):
        """
        return:
        groups = {
            <group name 1>: <group instance ids list>
            <group name 2>: [....]
        }
        """
        groups = {}
        for group_name, group_def in self._grouping_rules[self._fields.group_definitions].items():
            rule = EasyDict(group_def)
            if group_name not in groups:
                group_instance_ids = self._make_group(group_def)
                groups.update({group_name: group_instance_ids})
                print("divider: make group:{} , include {} instances".format(
                    group_name, len(group_instance_ids)))
        return groups

    def add_rule(self, rule_name, query_attr_names, db_attr_names):
        """
        TODO @dennis.liu: Need to implement.
        """
        pass

    def _make_group(self, rule):
        def is_filter(attrs):
            filter_attrs = self._get_rule_attrs(rule, self._fields.filter_attrs)
            return len(self.inner_join(attrs, filter_attrs)) > 0

        def is_in_union(attrs):
            union_attrs = self._get_rule_attrs(rule, self._fields.union_attrs)

            # all union
            if len(union_attrs) == 1 and union_attrs[0] == "all":
                return True
            # no union constraint
            if len(union_attrs) == 0:
                return True

            for attr in attrs:
                if attr in union_attrs:
                    return True
            else:
                return False

        inst_attrs = self._attr_container.instance_attributes

        instance_ids = []
        # for each instance
        for inst, attrs in inst_attrs.items():
            # filter first
            if not is_filter(attrs):
                # selet attributes in union attributes
                if is_in_union(attrs):
                    instance_ids.append(inst)
        return instance_ids

    def inner_join(self, setA, setB):
        return list(set(setA) & set(setB))

    def _get_rule_attrs(self, rule, field_name):
        rule_attrs = rule.get(field_name, "").split(',')
        if rule_attrs == [""]:
            return []
        else:
            return rule_attrs

