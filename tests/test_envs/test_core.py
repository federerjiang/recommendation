import unittest
import porise

class ArgumentEnv(porise.Env):
    calls = 0
    def __init__(self, arg):
        self.calls += 1
        self.arg = arg


class TestEnv(unittest.TestCase):

    def test_env_instantiation(self):
        env = ArgumentEnv('arg')
        assert env.arg == 'arg'
        assert env.calls == 1

if __name__ == '__main__':
    unittest.main()
    

