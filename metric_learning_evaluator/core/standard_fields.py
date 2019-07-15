"""
  Collection of standard fields
  Naming convention:
    - ObjectName + StandardFields
    - FolderName + StandardFields
    (Q: which is better?)
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))



class EvaluationStandardFields:
    """
      Standard declaration of evaluation objects
    """

    # ===== Evaluations =====

    # Retrival
    ranking = 'RankingEvaluation'

    # Ranking with Attributes Evaluation
    ranking_with_attrs = 'RankingWithAttributesEvaluation'

    mock = 'MockEvaluation'

    # FaceNet: Pair-wise evaluation
    facenet = 'FacenetEvaluation'

    checkout = 'CheckoutEvaluation'

    # Avaliable evaluation implementations
    classification = 'ClassificationEvaluation'

    geometric = 'GeometricEvaluation'

    # ===== Inner items =====
    sampling = 'sampling'
    metric = 'metric'
    distance_measure = 'distance_measure'
    attribute = 'attribute'
    option = 'option'

    # ===== Distance measure inner items =====
    function = 'function'
    threshold = 'threshold'
    start = 'start'
    end = 'end'
    step = 'step'


class ClassificationEvaluationStandardFields:
    top_1_hit_accuracy = 'top_1_hit_accuracy'
    top_k_hit_accuracy = 'top_k_hit_accuracy'
    mAP = 'mAP'


class RankingEvaluationStandardFields:
    start = 'start'
    end = 'end'
    step = 'step'
    top_k_hit_accuracy = 'top_k_hit_accuracy'
    mAP = 'mAP'
    sampling = 'sampling'


class ConfigParserStandardFields:
    pass


class ConfigStandardFields:
    """Items for Evaluation Configuration"""

    # Key of database
    database = 'database'

    # Types of warpper
    database_type = 'database_type'
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
    # group is the filtering condition
    group = 'group'

    # topics under `evaluation`
    metric = 'metric'
    attribute = 'attribute'
    option = 'option'
    distance_measure = 'distance_measure'
    sampling = 'sampling'

    # Following two are deprecated
    embedding_size = 'embedding_size'
    prob_size = 'prob_size'


class ApplicationStatusStandardFields:
    not_determined = 'not_determined'
    # evaluation applications
    evaluate_single_container = 'evaluate_single_container'
    evaluate_query_database = 'evaluate_query_database'
    # inference applications
    inference_feature_extraction = 'inference_feature_extraction'


class ImageObjectStandardFields:
    image_id = 'image_id'
    instance_id = 'instance_id'
    bounding_box = 'bounding_box'
    bounding_box_confidence = 'bounding_box_confidence'
    instance_feature = 'instance_feature'
    instance_label_id = 'instance_label_id'
    instance_label_name = 'instance_label_name'


class FeatureObjectStandardFields:
    embeddings = 'embeddings'
    probabilities = 'probabilities'
    label_ids = 'label_ids'
    label_names = 'label_names'
    instance_ids = 'instance_ids'
    filename_strings = 'filename_strings'
    super_labels = 'super_labels'


class ResultContainerStandardFields:
    pass


class EmbeddingContainerStandardFields:
    embeddings = 'embeddings'
    probabilities = 'probabilities'
    instance_ids = 'instance_ids'
    label_ids = 'label_ids'
    label_names = 'label_names'
    filename_strings = 'filename_strings'
    meta = 'meta'


class IndexAgentStandardFields:
    numpy_agent = 'Numpy'
    hnsw_agent = 'HNSW'


class MetricStandardFields:
    """
      Standard declaration of metric function, only functions defined here can be executed.
    """
    # === Metric Function and Items ===
    metric = 'metric'
    mAP = 'mAP'

    # Area Under Curve
    auc = 'AUC'
    area_under_curve = 'area_under_curve'

    # Equal Error Rate
    eer = 'EER'
    equal_error_rate = 'equal_error_rate'

    # False Accepted Rate: FAR = TA(d)/P_same
    far = 'FAR'
    false_accept_rate = 'false_accept_rate'

    # Validation Rate: VAR = FA(d)/P_diff
    var = 'VAR'
    validation_rate = 'validation_rate'

    accuracy = 'accuracy'
    # Top k accuracy
    top_k_accuracy = 'top_k_accuracy'
    top_k_hit_accuracy = 'top_k_hit_accuracy'
    
    pair_sampling = 'pair_sampling'
    ranking = 'ranking'

    # deprecated
    distance_function = 'distance_function'
    distance_threshold = 'distance_threshold'
    path_pairlist = 'path_pairlist'
    sample_method = 'sample_method'
    sample_ratio = 'sample_ratio'
    class_sample_method = 'class_sample_method'
    ratio_of_class = 'ratio_of_class'
    ratio_of_image_per_class = 'ratio_of_image_per_class'

    # === Parameters & Internal Keys ===
    true_positive = 'true_positive'
    false_positive = 'false_positive'
    true_negative = 'true_negative'
    false_negative = 'false_negative'
    true_positive_rate = 'true_positive_rate'
    false_positive_rate = 'false_positive_rate'



class QueryDatabaseStandardFields:
    datasetbackbone = 'DatasetBackbone'
    zeus = 'Zeus'
    json = 'Json'
    # built-in sqlite3 attribute table
    native = 'Native'


class AttributeStandardFields(object):
    """Collection of Known Attributes."""
    # Special keywords:
    attr_key = 'attribute'
    attribute = 'attribute'
    all_attributes = 'all_attributes'
    all_classes = 'all_classes'
    # none attributes are given, use class instead.
    none = 'none'
    All = 'all'
    # TODO @kv:
    per_category = 'per_category'

    # General attribute types:
    category = 'category'
    supercategory = 'supercategory'

    # Preserved keywords
    #seen = 'seen'
    #unseen = 'unseen'
    query = 'query'
    database = 'database'
    cross_reference = 'cross_reference'
    filtering = 'filtering'
    group = 'group'

    # Customized for checkout
    seen_to_seen = 'seen_to_seen'
    unseen_to_unseen = 'unseen_to_unseen'
    seen_to_total = 'seen_to_total'
    unseen_to_total = 'unseen_to_total'