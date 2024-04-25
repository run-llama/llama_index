import os

import pytest

from llama_index.llms.nvidia import NVIDIA

from contextlib import contextmanager
from typing import Any, Generator


@contextmanager
def no_env_var(var: str) -> Generator[None, None, None]:
    try:
        if val := os.environ.get(var, None):
            del os.environ[var]
        yield
    finally:
        if val:
            os.environ[var] = val


@pytest.mark.integration()
@pytest.mark.parametrize("param", ["nvidia_api_key", "api_key"])
def test_additional_kwargs_success(chat_model: str, mode: dict, param: str) -> None:
    api_key = os.environ.get("NVIDIA_API_KEY")
    with no_env_var("NVIDIA_API_KEY"):
        client = NVIDIA(chat_model, **{param: api_key}).mode(**mode)
        assert client.complete(
            "Hello, world!", 
            stop=["cat", "Cats"],
            seed=42,
            frequency_penalty=0.5,
            presence_penalty=0.5,
        ).text


@pytest.mark.integration()
@pytest.mark.parametrize("param", ["nvidia_api_key", "api_key"])
def test_additional_kwargs_wrong_dtype(chat_model: str, mode: dict, param: str) -> None:
    api_key = os.environ.get("NVIDIA_API_KEY")
    with no_env_var("NVIDIA_API_KEY"):
        client = NVIDIA(chat_model, **{param: api_key}).mode(**mode)
        with pytest.raises(Exception) as exc_info:
            client.complete(
                "Hello, world!", 
                frequency_penalty="fish",
            ).text
        message = str(exc_info.value)
        assert "400" in message


@pytest.mark.integration()
@pytest.mark.parametrize("param", ["nvidia_api_key", "api_key"])
def test_additional_kwargs_wrong_dtype(chat_model: str, mode: dict, param: str) -> None:
    api_key = os.environ.get("NVIDIA_API_KEY")
    with no_env_var("NVIDIA_API_KEY"):
        client = NVIDIA(chat_model, **{param: api_key}).mode(**mode)
        with pytest.raises(Exception) as exc_info:
            client.complete(
                "Hello, world!", 
                cats="cats",
            ).text
        message = str(exc_info.value)
        assert "unexpected keyword" in message