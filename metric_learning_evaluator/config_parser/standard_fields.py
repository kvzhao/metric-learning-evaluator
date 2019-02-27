
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

class ConfigStandardFields(object):
    """Items for Evaluation Configuration"""

    # Key of database
    database = 'database'
    # Types of database
    datasetbackbone = 'DatasetBackbone'
    zeus = 'Zeus'
    none = 'None'
    # System-related configs
    container_size = 'container_size'
    chosen_evaluations = 'chosen_evaluations'

    # Keywords of Evaluation & Attributes
    evaluation = 'evaluation'
    evaluation_options = 'evaluation_options'

    # topics under `evaluation`
    metric = 'metric'
    attribute = 'attribute'
    option = 'option'
    distance_measure = 'distance_measure'
    sampling = 'sampling'

    # Following two are deprecated
    embedding_size = 'embedding_size'
    logit_size = 'logit_size'