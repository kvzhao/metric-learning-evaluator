
import numpy as np

from evaluation_base import EmbeddingDatum


embedding_type = [('image_id', int), ('label_id', int), ('embedding', np.array)]
feat1 = EmbeddingDatum(122, 1, np.array([1,2,4,5]))
feat2 = EmbeddingDatum(121, 9, np.array([3,2,4,5]))
feat3 = EmbeddingDatum(125, 2, np.array([4,2,4,5]))


feat_tuple = (122, 1, np.array([1,2,4,5]))

features = np.array([feat_tuple], dtype=embedding_type)
print (features)
