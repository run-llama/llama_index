# pants requires this import to recognize the dep
import pytest_asyncio  # noqa: F401

import pytest
import os


from typing import Generator


# this fixture is used to mask the NVIDIA_API_KEY environment variable and restore it
# after the test. it also returns the value of the NVIDIA_API_KEY environment variable
# before it was masked so that it can be used in the test.
@pytest.fixture()
def masked_env_var() -> Generator[str, None, None]:
    var = "NVIDIA_API_KEY"
    # Save the current value of the environment variable, if it exists
    val = os.environ.get(var, None)

    # Remove the environment variable to simulate it being masked during the test
    if val is not None:
        del os.environ[var]

    try:
        # Yield the original value so it can be used in the test
        yield val
    finally:
        # Restore the original environment variable if it was set
        if val is not None:
            os.environ[var] = val
        else:
            # If the variable was not originally set, ensure it's removed
            if var in os.environ:
                del os.environ[var]


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
    from llama_index.llms.nvidia import NVIDIA
    from llama_index.llms.nvidia.base import DEFAULT_MODEL

    mode = get_mode(metafunc.config)

    if "chat_model" in metafunc.fixturenames:
        models = [DEFAULT_MODEL]
        if model := metafunc.config.getoption("--model-id"):
            models = [model]
        elif metafunc.config.getoption("--all-models"):
            models = [model.id for model in NVIDIA(**mode).available_models]
        metafunc.parametrize("chat_model", models, ids=models)


@pytest.fixture()
def mode(request: pytest.FixtureRequest) -> dict:
    return get_mode(request.config)
