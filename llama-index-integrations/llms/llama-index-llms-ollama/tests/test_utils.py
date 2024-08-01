from llama_index.llms.ollama.base import get_additional_kwargs


def test_get_additional_kwargs():
    response = {"key1": "value1", "key2": "value2", "exclude_me": "value3"}
    exclude = ("exclude_me", "exclude_me_too")

    expected = {"key1": "value1", "key2": "value2"}

    actual = get_additional_kwargs(response, exclude)

    assert actual == expected
