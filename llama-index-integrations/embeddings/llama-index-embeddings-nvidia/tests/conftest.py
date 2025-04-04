# pants requires this import to recognize the dep
import pytest_asyncio  # noqa: F401

import pytest
import os

from llama_index.embeddings.nvidia import NVIDIAEmbedding as Interface
from llama_index.embeddings.nvidia.base import DEFAULT_MODEL

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


@pytest.fixture(params=[Interface])
def public_class(request: pytest.FixtureRequest) -> type:
    return request.param


def pytest_collection_modifyitems(config, items):
    if "NVIDIA_API_KEY" not in os.environ:
        skip_marker = pytest.mark.skip(
            reason="requires NVIDIA_API_KEY environment variable or --nim-endpoint option"
        )
        for item in items:
            if "integration" in item.keywords and not config.getoption(
                "--nim-endpoint"
            ):
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


def get_mode(config: pytest.Config) -> dict:
    nim_endpoint = config.getoption("--nim-endpoint")
    if nim_endpoint:
        return {"base_url": nim_endpoint}
    return {}


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    mode = get_mode(metafunc.config)

    if "model" in metafunc.fixturenames:
        models = [DEFAULT_MODEL]
        if model := metafunc.config.getoption("--model-id"):
            models = [model]
        elif metafunc.config.getoption("--all-models"):
            models = [model.id for model in Interface(**mode).available_models]
        metafunc.parametrize("model", models, ids=models)


@pytest.fixture()
def mode(request: pytest.FixtureRequest) -> dict:
    return get_mode(request.config)
