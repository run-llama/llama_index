from ollama import GenerateResponse

from llama_index.llms.ollama.base import get_additional_kwargs


def test_get_additional_kwargs():
    response = {"key1": "value1", "key2": "value2", "exclude_me": "value3"}
    exclude = ("exclude_me", "exclude_me_too")

    expected = {"key1": "value1", "key2": "value2"}

    actual = get_additional_kwargs(response, exclude)

    assert actual == expected


def test_get_additional_kwargs_with_ollama_response_object():
    # ollama >= 0.4 returns pydantic models instead of plain dicts;
    # see https://github.com/run-llama/llama_index/issues/17105
    response = GenerateResponse(
        model="llava",
        created_at="2024-11-29T00:00:00Z",
        response="a plate of fried chicken",
        done=True,
        eval_count=42,
    )

    actual = get_additional_kwargs(response, ("response",))

    assert actual["model"] == "llava"
    assert actual["eval_count"] == 42
    assert "response" not in actual
