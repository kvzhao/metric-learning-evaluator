import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))



class ApplicationStatusStandardFields:
    not_determined = 'not_determined'

    # evaluation applications
    evaluate_single_container = 'evaluate_single_container'
    evaluate_query_database = 'evaluate_query_database'


    # inference applications
    inference_feature_extraction = 'inference_feature_extraction'