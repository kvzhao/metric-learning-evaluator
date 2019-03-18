
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

class EvaluationStandardFields(object):
    """
      Standard declaration of evaluation objects
    """

    # ===== Evaluations =====

    # Retrival
    ranking = 'RankingEvaluation'

    checkout = 'CheckoutEvaluation'

    mock = 'MockEvaluation'

    # FaceNet: Pair-wise evaluation
    facenet = 'FacenetEvaluation'

    # Avaliable evaluation implementations
    classification = 'ClassificationEvaluation'

    # ===== Inner items =====
    sampling = 'sampling'
    metric = 'metric'
    distance_measure = 'distance_measure'
    attribute = 'attribute'
    option = 'option'