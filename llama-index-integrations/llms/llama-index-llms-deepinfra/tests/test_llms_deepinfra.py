from llama_index.llms.deepinfra import DeepInfraLLM
from llama_index.llms.deepinfra.utils import maybe_decode_sse_data
from llama_index.core.llms.llm import LLM


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in DeepInfraLLM.__mro__]
    assert LLM.__name__ in names_of_base_classes


def test_deepinfra_llm_class():
    model = DeepInfraLLM(api_key="test")
    assert isinstance(model, LLM)


def test_deepinfra_llm_from_env():
    import os

    os.environ["DEEPINFRA_API_TOKEN"] = "test"
    model = DeepInfraLLM()
    assert isinstance(model, LLM)


def test_maybe_decode_sse_data_only_strips_the_prefix():
    assert maybe_decode_sse_data(b'data: {"text": "data"}') == {"text": "data"}
    # `str.strip("data: ")` would also eat the leading "t" of a bare JSON literal
    assert maybe_decode_sse_data(b"data: true") is True
    assert maybe_decode_sse_data(b"data: [DONE]") is None
    assert maybe_decode_sse_data(b"not an sse line") is None
