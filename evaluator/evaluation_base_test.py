
import numpy as np

from evaluation_base import EmbeddingContainer, AttributeContainer

# Evaluator Builder
# Customized Evaluator
# Evaluator Base

batch = 32
embed_size = 1024

features = np.random.rand(batch, embed_size)

embed_container = EmbeddingContainer(embedding_size=embed_size)

for source_id, feat in enumerate(features):
    embed_container.add(source_id, 0, feat)

print(embed_container.embeddings)
print(embed_container.embeddings.shape)

print (embed_container.get_embedding_by_image_ids([0,5,6]))
print (embed_container.get_embedding_by_image_ids([0,5,6]).shape)

print (embed_container.get_label_by_image_ids([0]))
print (embed_container.get_label_by_image_ids(0))


attr_container = AttributeContainer()

attr_container.add("color.red", [2, 5, 6, 8, 10])
attr_container.add("color.blue", [1, 2, 4, 7, 9])
attr_container.add("color.red", [11, 13, 17])
attr_container.add("color.blue", 19)

print (attr_container.groups)


for k, v in attr_container.groups.items():
    print (k, v)

print (attr_container.attr_lookup)
print (attr_container.attr_name_lookup)