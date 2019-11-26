
import numpy as np
from data_container import EmbeddingContainer
from data_container import AttributeContainer
from data_container import ResultContainer

import unittest
from collections import defaultdict


class TestEmbeddingContainer(unittest.TestCase):
    def test_init(self):
        container_size = 32
        embed_size = 1024
        logit_size = 0
        embed_container = EmbeddingContainer(embedding_size=embed_size,
            logit_size=logit_size, container_size=container_size)

    def test_add_function_and_internals(self):
        """
            Push mock data and fetch. Check pushed data and answers are consistent.
        """

        batch_size = 512
        container_size = 1000
        embed_size = 1024
        logit_size = 128
        
        embedding_container = EmbeddingContainer(embedding_size=embed_size,
            logit_size=logit_size, container_size=container_size)
        mock_embeddings = np.random.rand(batch_size, embed_size)
        mock_logits = np.random.rand(batch_size, logit_size)

        correct_image_ids = []
        correct_label_ids = []
        image_id_to_label_id = {}
        label_id_to_image_ids = defaultdict(list)

        for image_id, (embedding, logit) in enumerate(zip(mock_embeddings, mock_logits)):
            label_id = image_id % 10
            correct_label_ids.append(label_id)
            correct_image_ids.append(image_id)
            image_id_to_label_id[image_id] = label_id
            label_id_to_image_ids[label_id].append(image_id)
            embedding_container.add(image_id, label_id, embedding, logit)

        self.assertEqual(embedding_container.embeddings.shape[0],
                         embedding_container.logits.shape[0],
                         'Internal numpy matrix: logit and embedding are not with the same container size.')
        self.assertTrue(np.allclose(mock_embeddings, embedding_container.embeddings), 'Mismatching embeddings.')
        self.assertTrue(np.allclose(mock_logits, embedding_container.logits), 'Mismatching logits.')
        self.assertEqual(batch_size, embedding_container.counts, 'Embeddings not all pushed in container.')

        for label_id, image_ids in label_id_to_image_ids.items():
            self.assertEqual(image_ids,
                embedding_container.get_image_ids_by_label(label_id),
                'Feteched image ids are not equal.')

        for image_id, label_id in image_id_to_label_id.items():
            returned_label_id = embedding_container.get_label_by_image_ids(image_id)
            self.assertEqual(returned_label_id, label_id, 'Feteched label ids are not equal.')

        self.assertEqual(correct_image_ids, embedding_container.image_ids,
                         'Fetched image_ids are not equal to image_ids.')

class TestAttributeContainer(unittest.TestCase):

    def test_add_function_and_internals(self):
        """
            Push mock data and fetch. Check pushed data and answers are consistent.
        """
        attribute_container = AttributeContainer()

        mock_database = {
            1: ["Color.Red", "Shape.Bottle", "Pose.isFront"],
            2: ["Color.Red", "Shape.Can", "Pose.isBack"],
            3: ["Color.Blue", "Shape.Bottle"],
        }
        # prepare targets
        all_image_ids = list(mock_database.keys())
        all_attributes = []
        for _, _attr in mock_database.items():
            all_attributes.extend(_attr)
        all_attributes = list(set(all_attributes))

        correct_groupings = {
            'Color.Red': [1, 2],
            'Color.Blue': [3],
            'Shape.Can': [2],
            'Shape.Bottle': [1, 3],
            'Pose.isFront': [1],
            'Pose.isBack': [2],
        }

        # push and assert
        for _img_id, _attrs in mock_database.items():
            attribute_container.add(_img_id, _attrs)
        
        self.assertEqual(correct_groupings,
                         attribute_container.groups,
                         '')


class TestResultConatiner(unittest.TestCase):

    def test_add(self):
        pass

if __name__ == '__main__':
    unittest.main()