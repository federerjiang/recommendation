import unittest
from porise.envs.synthetic.linear_env import LinearEnv
from porise.envs.utils import seeding, error 


class TestEnv(unittest.TestCase):

    def test_env_instantiation(self):
        env = LinearEnv(n_arm=3, feature_dim=10, max_steps=100)
        _, reward, done, info = env.step(0)
        print(done, reward, info, env.steps_beyond_done)
        while not done:
            _, reward, done, info = env.step(env.action_space.sample())
            print(done, reward, info, env.steps_beyond_done)
        assert env.steps_beyond_done == 0
    
    def test_done(self):
        max_steps = 100
        env = LinearEnv(n_arm=3, feature_dim=10, max_steps=max_steps)
        while max_steps > 0:
            _, _, done, _ = env.step(env.action_space.sample())
            max_steps -= 1
        assert done == True
            
            
if __name__ == '__main__':
    unittest.main()