
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

class ConfigStandardFields(object):
    """Items for Evaluation Configuration"""

    # Key of database
    database = 'database'

    # Types of warpper
    database_type = 'database_type'
    datasetbackbone = 'DatasetBackbone'
    zeus = 'Zeus'
    json = 'Json'
    none = 'None'
    native = 'Native'

    # Config of wrapper
    database_config = 'database_config'

    # Agent options
    index_agent = 'index_agent'
    numpy_agent = 'Numpy'
    hnsw_agent = 'HNSW'

    # System-related configs
    container_size = 'container_size'
    chosen_evaluations = 'chosen_evaluations'

    # Keywords of Evaluation & Attributes
    evaluation = 'evaluation'
    evaluation_options = 'evaluation_options'

    cross_reference = 'cross_reference'
    filtering = 'filtering'

    # topics under `evaluation`
    metric = 'metric'
    attribute = 'attribute'
    option = 'option'
    distance_measure = 'distance_measure'
    sampling = 'sampling'

    # Following two are deprecated
    embedding_size = 'embedding_size'
    logit_size = 'logit_size'
    prob_size = 'prob_size'