import pytest
import os


def pytest_collection_modifyitems(config, items):
    if "NVIDIA_API_KEY" not in os.environ:
        skip_marker = pytest.mark.skip(
            reason="requires NVIDIA_API_KEY environment variable"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_marker)
