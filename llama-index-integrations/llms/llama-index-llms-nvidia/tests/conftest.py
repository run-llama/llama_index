import pytest
import os

from llama_index.llms.nvidia.base import DEFAULT_PLAYGROUND_MODEL
from llama_index.llms.nvidia.utils import API_CATALOG_MODELS


def pytest_collection_modifyitems(config, items):
    if "NVIDIA_API_KEY" not in os.environ:
        skip_marker = pytest.mark.skip(
            reason="requires NVIDIA_API_KEY environment variable"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_marker)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--all-models",
        action="store_true",
        help="Run tests across all models",
    )
    parser.addoption(
        "--model-id",
        action="store",
        help="Run tests for a specific chat model",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "chat_model" in metafunc.fixturenames:
        models = [DEFAULT_PLAYGROUND_MODEL]
        if model := metafunc.config.getoption("--model-id"):
            models = [model]
        elif metafunc.config.getoption("--all-models"):
            models = list(API_CATALOG_MODELS.keys())
        metafunc.parametrize("chat_model", models, ids=models)
