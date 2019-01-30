
import unittest

from config_parser import ConfigParser

path_of_the_config = 'eval_config.yml'

class TestConfigParser(unittest.TestCase):
    def test_has_attributes(self):
        config = ConfigParser(path_of_the_config)
        print (config.evaluation_names)
        self.assertTrue(config.has_attribute('RankingEvaluation'))
        self.assertFalse(config.has_attribute('MockEvaluation'))
        print (config.get_per_eval_config('EmptyEvaluation'))
        print (config.get_per_eval_config('NotExist'))

if __name__ == '__main__':
    unittest.main()