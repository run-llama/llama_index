import pytest
import os

from llama_index.multi_modal_llms.nvidia import NVIDIAMultiModal as Interface
from llama_index.multi_modal_llms.nvidia.utils import DEFAULT_MODEL

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


def pytest_collection_modifyitems(config, items):
    if "NVIDIA_API_KEY" not in os.environ:
        skip_marker = pytest.mark.skip(
            reason="requires NVIDIA_API_KEY environment variable"
        )
        for item in items:
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
    parser.addoption(
        "--nim-endpoint",
        type=str,
        help="Run tests using NIM mode",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "vlm_model" in metafunc.fixturenames:
        models = [DEFAULT_MODEL]
        if model := metafunc.config.getoption("--model-id"):
            models = [model]
        elif metafunc.config.getoption("--all-models"):
            models = [model.id for model in Interface().available_models]
        metafunc.parametrize("vlm_model", models, ids=models)
