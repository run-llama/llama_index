import os
import pytest
from typing import Generator


@pytest.fixture(autouse=True)
def clean_env() -> Generator[None, None, None]:
    """Clean up environment variables before and after each test."""
    # Store original environment variables
    original_env = {}
    for key in ["INFERENCE_KEY", "INFERENCE_URL", "INFERENCE_MODEL_ID"]:
        if key in os.environ:
            original_env[key] = os.environ[key]
            del os.environ[key]

    yield

    # Restore original environment variables
    for key, value in original_env.items():
        os.environ[key] = value


@pytest.fixture()
def masked_env_var() -> str:
    """Return a masked environment variable for testing."""
    return "test-key-12345"
