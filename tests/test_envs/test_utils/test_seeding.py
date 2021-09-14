import unittest

from porise.envs.utils import seeding, error 


class TestSeed(unittest.TestCase):

    def test_invalid_seeds(self):
        for seed in [-1, 'test']:
            try:
                seeding.np_random(seed)
            except error.Error:
                pass
            else:
                assert False, 'Invalid seed {} passed validation'.format(seed)
    
    def test_valid_seeds(self):
        for seed in [0, 1]:
            random, seed1 = seeding.np_random(seed)
            assert seed == seed1


if __name__ == '__main__':
    unittest.main()
    

