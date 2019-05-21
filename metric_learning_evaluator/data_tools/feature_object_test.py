
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from metric_learning_evaluator.data_tools.feature_object import FeatureObjectBase
from metric_learning_evaluator.data_tools.feature_object import FeatureObject

feature_object = FeatureObject()

# TODO: change path
feature_object.load('extracted_embeddings_facenet-batch512')

#print (feature_object.features.shape)

feature_object.features = np.asarray([1,2,3])
print (feature_object.features)

print (feature_object.features.shape)

feature_object.save('test_numpy_dump')
#

print (feature_object)