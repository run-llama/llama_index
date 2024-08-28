import pytest
import os


from typing import Generator


# this fixture is used to mask the NVIDIA_API_KEY environment variable and restore it
# after the test. it also returns the value of the NVIDIA_API_KEY environment variable
# before it was masked so that it can be used in the test.
@pytest.fixture()
def masked_env_var() -> Generator[str, None, None]:
    var = "NVIDIA_API_KEY"
    try:
        if val := os.environ.get(var, None):
            del os.environ[var]
        yield val
    finally:
        if val:
            os.environ[var] = val
