"""Standard Fields
  Standard fields define all keys would be used in the evaluator system.

  Naming convention:
    - ObjectName + StandardFields
    - FolderName + StandardFields
    (Q: which is better?)
  @kv
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
    # FaceNet: Pair-wise evaluation
    facenet = 'FacenetEvaluation'
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


class FacenetEvaluationStandardFields:
    """Define fields used only in Facenet evaluation
        which may assign in `option` section in config.
    """
    # pair dict
    pairA = 'pairA'
    pairB = 'pairB'
    is_same = 'is_same'
    path_pairlist = 'path_pairlist'
    num_maximum_pairs = 'num_maximum_pairs'
    num_of_pairs = 'num_of_pairs'
    # used for distance threshold
    start = 'start'
    end = 'end'
    step = 'step'
    # sampling options
    sample_method = 'sample_method'
    sample_ratio = 'sample_ratio'
    ratio_of_class = 'ratio_of_class'
    ratio_of_instance_per_class = 'ratio_of_instance_per_class'
    num_of_instance_per_class = 'num_of_instance_per_class'
    # sampling methods
    class_sample_method = 'class_sample_method'
    random_sample = 'random_sample'
    amount_weighted = 'amount_weighted'
    amount_inverse_weighted = 'amount_inverse_weighted'


class ConfigParserStandardFields:
    pass


class ConfigStandardFields:
    """Items for Evaluation Configuration"""

    # === Evaluation Configs ===
    # Key of database
    database = 'database'

    # Types of warpper
    database_type = 'database_type'
    csv = 'CSV'

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

    embedding_container_capacity = 'embedding_container_capacity'


class ApplicationStatusStandardFields:
    not_determined = 'not_determined'
    # evaluation applications
    evaluate_single_container = 'evaluate_single_container'
    evaluate_query_anchor = 'evaluate_query_anchor'
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


class ImageRetrievalStandardFields:
    detector_settings = 'detector_settings'
    extractor_settings = 'extractor_settings'
    num_classes = 'num_classes'
    model_path = 'model_path'
    labelmap_path = 'labelmap_path'
    image_size = 'image_size'
    embedding_size = 'embedding_size'
    database_path = 'database_path'


class DetectorStandardFields:
    num_detections = 'num_detections'
    detection_classes = 'detection_classes'
    detection_boxes = 'detection_boxes'
    detection_scores = 'detection_scores'


class FeatureObjectStandardFields:
    embeddings = 'embeddings'
    probabilities = 'probabilities'
    label_ids = 'label_ids'
    label_names = 'label_names'
    instance_ids = 'instance_ids'
    filename_strings = 'filename_strings'
    super_labels = 'super_labels'
    bounding_boxes = 'bounding_boxes'
    landmarks = 'landmarks'


class ResultContainerStandardFields:
    pass


class EmbeddingContainerStandardFields:
    embeddings = 'embeddings'
    probabilities = 'probabilities'
    # Same as CsvFile columns
    instance_ids = 'instance_id'
    label_ids = 'label_id'
    label_names = 'label_name'
    filename_strings = 'filename_string'
    meta = 'meta'


class AttributeTableStandardFields:
    # And CsvFile column
    instance_id = 'instance_id'
    label_id = 'label_id'
    label_name = 'label_name'
    image_path = 'image_path'
    filename_string = 'filename_string'


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
    mean_accuracy = 'mean_accuracy'
    mean_validation_rate = 'mean_validation_rate'
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

    k_fold = 'k_fold'


class QueryDatabaseStandardFields:
    # Support only csv
    csv = 'CSV'


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
    per_category = 'per_category'

    # General attribute types:
    category = 'category'
    supercategory = 'supercategory'

    # Preserved keywords
    query = 'query'
    database = 'database'
    cross_reference = 'cross_reference'
    filtering = 'filtering'
    group = 'group'


class InterpreterStandardField:
    LOAD_LIST = 'LOAD_LIST'
    STORE_NAME = 'STORE_NAME'
    LOAD_NAME = 'LOAD_NAME'
    JOIN = 'JOIN'
    REMOVE = 'REMOVE'
    PRINT = 'PRINT'
    instructions = 'instructions'
    values = 'values'
    names = 'names'

    AND = 'AND'
    NOT = 'NOT'
    OR = 'OR'
    XOR = 'XOR'
    SYMBOL = 'SYMBOL'
    LPAREN = '('
    RPAREN = ')'
    EOF = 'EOF'


class SampleStrategyStandardFields:
    # sampling
    sample_method = 'sample_method'
    class_sample_method = 'class_sample_method'
    instance_sample_method = 'instance_sample_method'

    # mothods
    uniform = 'uniform'
    all_class = 'all_class'
    all_instance = 'all_instance'
    instance_number_weighted = 'instance_number_weighted'
    instance_number_inverse_weighted = 'instance_number_inverse_weighted'

    # sampling options
    sample_ratio = 'sample_ratio'

    ratio_of_instance = 'ratio_of_instance'
    ratio_of_sampled_class = 'ratio_of_sampled_class'
    ratio_of_instance_per_class = 'ratio_of_instance_per_class'

    num_of_sampled_class = 'num_of_sampled_class'
    num_of_sampled_instance_per_class = 'num_of_sampled_instance_per_class'

    num_of_db_class = 'num_of_db_class'
    num_of_db_instance = 'num_of_db_instance'
    num_of_db_instance_per_class = 'num_of_db_instance_per_class'
    num_of_query_class = 'num_of_query_class'
    num_of_query_instance_per_class = 'num_of_query_instance_per_class'
    maximum_of_sampled_data = 'maximum_of_sampled_data'

    # pair
    is_same = 'is_same'
    num_of_pairs = 'num_of_pairs'
    ratio_of_positive_pair = 'ratio_of_positive_pair'
    ratio_of_negative_pair = 'ratio_of_negative_pair'
    pair_A = 'pair_A'
    pair_B = 'pair_B'
    pair_A_label = 'pair_A_label'
    pair_B_label = 'pair_B_label'

    # ranking
    sampled_instance_ids = 'sampled_instance_ids'
    sampled_label_ids = 'sampled_label_ids'
    query_instance_ids = 'query_instance_ids'
    query_label_ids = 'query_label_ids'
    db_instance_ids = 'db_instance_ids'
    db_label_ids = 'db_label_ids'
    database_instance_ids = 'database_instance_ids'
    database_label_ids = 'database_label_ids'
