import unittest
from porise.envs.synthetic.bernoulli_env import BernoulliEnv
from porise.envs.utils import seeding, error 


class TestEnv(unittest.TestCase):

    def test_env_instantiation(self):
        env = BernoulliEnv(n_arm=3, arm_probs=[0.1, 0.3, 0.6], max_steps=100)
        _, _, done, _ = env.step(0)
        while not done:
            _, reward, done, _ = env.step(env.action_space.sample())
            print(done, reward)
    
    def test_invalid_arm_probs(self):
        try: 
            env = BernoulliEnv(n_arm=3, arm_probs=[0.1, 0.3, 0.7], max_steps=100)
        except AssertionError as error:
            pass
        else:
            assert False, 'Sum of arm probs {} is not equal to 1.0, but still passed validation'.format(env.arm_probs)
            
            
if __name__ == '__main__':
    unittest.main()