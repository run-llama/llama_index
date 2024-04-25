import pytest
import os

from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.embeddings.nvidia.base import DEFAULT_MODEL

from typing import Generator

from contextlib import contextmanager


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
        return {"mode": "nim", "base_url": nim_endpoint}
    return {}


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    mode = get_mode(metafunc.config)

    if "model" in metafunc.fixturenames:
        models = [DEFAULT_MODEL]
        if model := metafunc.config.getoption("--model-id"):
            models = [model]
        elif metafunc.config.getoption("--all-models"):
            models = [
                model.id for model in NVIDIAEmbedding().mode(**mode).available_models
            ]
        metafunc.parametrize("model", models, ids=models)


@pytest.fixture()
def mode(request: pytest.FixtureRequest) -> dict:
    return get_mode(request.config)


@contextmanager
def no_env_var(var: str) -> Generator[None, None, None]:
    try:
        if val := os.environ.get(var, None):
            del os.environ[var]
        yield
    finally:
        if val:
            os.environ[var] = val
