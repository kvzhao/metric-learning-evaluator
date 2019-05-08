"""Label map utility functions."""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import json

def read_json_labelmap(json_path):

    try:
        with open(json_path, 'r') as fp:
            label_map = json.load(fp)
    except:
        raise IOError('Labelmap: {} can not be loaded.'.format(json_path))

    # Check the structure: dict of dict
    # if not (dict: str-int mapping or int-str mapping), convert automatically
    # TODO @kv

    return label_map