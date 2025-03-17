import pytest

from llama_index.llms.nvidia import NVIDIA


@pytest.mark.integration
def test_additional_kwargs_success(chat_model: str, mode: dict) -> None:
    client = NVIDIA(chat_model, **mode)
    assert client.complete(
        "Hello, world!",
        stop=["cat", "Cats"],
        seed=42,
        frequency_penalty=0.5,
        presence_penalty=0.5,
    ).text


@pytest.mark.integration
def test_additional_kwargs_wrong_dtype(chat_model: str, mode: dict) -> None:
    client = NVIDIA(chat_model, **mode)
    with pytest.raises(Exception) as exc_info:
        client.complete(
            "Hello, world!",
            frequency_penalty="fish",
        ).text
    message = str(exc_info.value)
    assert "400" in message


@pytest.mark.integration
def test_additional_kwargs_wrong_dtype(chat_model: str, mode: dict) -> None:
    client = NVIDIA(chat_model, **mode)
    with pytest.raises(Exception) as exc_info:
        client.complete(
            "Hello, world!",
            cats="cats",
        ).text
    message = str(exc_info.value)
    assert "unexpected keyword" in message
