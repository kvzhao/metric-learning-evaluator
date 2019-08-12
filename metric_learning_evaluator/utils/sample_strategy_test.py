
import unittest
from sample_strategy import SampleStrategy


# Setting global random seed to
# make sure sampling is same value every time
import random
import numpy as np
random.seed(4065)
np.random.seed(4065)


class SampleStrategyTestCases(unittest.TestCase):

    def test_undefine_sampling(self):
        print("---test_undefine_sampling---")
        instance_ids = [1, 2, 3, 4, 5]
        label_ids = [1, 1, 2, 2, 1]

        sampler = SampleStrategy(instance_ids, label_ids)
        samped_ids_dict = sampler.sample('A', 'A', 2, 2, 4)
        sampled_label_ids = samped_ids_dict["sampled_label_ids"]
        sampled_instance_ids = samped_ids_dict["sampled_instance_ids"]
        print("instance_ids:{}".format(instance_ids))
        print("label_ids:{}".format(label_ids))
        print("sampled_label_ids:{}".format(sampled_label_ids))
        print("sampled_instance_ids:{}".format(sampled_instance_ids))
        self.assertAlmostEqual(sampled_label_ids, [2, 2, 1, 1])
        self.assertAlmostEqual(sampled_instance_ids, [3, 4, 2, 1])

    def test_class_uniform_sampler(self):
        print("---class_uniform_sampler---")
        instance_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        label_ids = [1, 1, 2, 2, 1, 2, 2, 1]
        print("instance_ids:{}".format(instance_ids))
        print("label_ids:{}".format(label_ids))
        sampler = SampleStrategy(instance_ids, label_ids)
        samped_ids_dict = sampler.class_uniform_sampler(num_of_sampled_class=2,
                                                        num_of_db_instance_per_class=1,
                                                        num_of_query_instance_per_class=2,
                                                        maximum_of_sampled_data=None)
        sampled_db = samped_ids_dict[0]
        sampled_label_ids = sampled_db["sampled_label_ids"]
        sampled_instance_ids = sampled_db["sampled_instance_ids"]
        print("sampled_label_ids:{}".format(sampled_label_ids))
        print("sampled_instance_ids:{}".format(sampled_instance_ids))
        self.assertAlmostEqual(sampled_label_ids, [1, 2])
        self.assertAlmostEqual(sampled_instance_ids, [1, 3])

        sampled_query = samped_ids_dict[1]
        sampled_label_ids = sampled_query["sampled_label_ids"]
        sampled_instance_ids = sampled_query["sampled_instance_ids"]
        print("sampled_label_ids:{}".format(sampled_label_ids))
        print("sampled_instance_ids:{}".format(sampled_instance_ids))
        self.assertAlmostEqual(sampled_label_ids, [1, 1, 2, 2])
        self.assertAlmostEqual(sampled_instance_ids, [2, 8, 7, 6])


if __name__ == "__main__":
    unittest.main()
