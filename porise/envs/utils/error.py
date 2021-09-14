# referred from https://github.com/openai/gym/blob/master/gym/error.py

import sys

class Error(Exception):
    pass

# Local errors

class Unregistered(Error):
    """Raised when the user requests an item from the registry that does
    not actually exist.
    """
    pass

class UnregisteredEnv(Unregistered):
    """Raised when the user requests an env from the registry that does
    not actually exist.
    """
    pass

class UnregisteredBenchmark(Unregistered):
    """Raised when the user requests an env from the registry that does
    not actually exist.
    """
    pass

class DeprecatedEnv(Error):
    """Raised when the user requests an env from the registry with an
    older version number than the latest env with the same name.
    """
    pass