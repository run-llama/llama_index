import importlib.util

import pytest

has_openvino_genai = importlib.util.find_spec("openvino_genai") is not None


@pytest.mark.skipif(
    not has_openvino_genai,
    reason="openvino-genai not installed",
)
def test_llm_class():
    from llama_index.core.llms.custom import CustomLLM
    from llama_index.llms.openvino_genai import OpenVINOGenAILLM

    names_of_base_classes = [b.__name__ for b in OpenVINOGenAILLM.__mro__]
    assert CustomLLM.__name__ in names_of_base_classes


@pytest.mark.skipif(
    not has_openvino_genai,
    reason="openvino-genai not installed",
)
def test_streamer_uses_write_method():
    """Verify that the streamer classes define a 'write' method (not deprecated 'put')."""
    import inspect

    from llama_index.llms.openvino_genai.base import OpenVINOGenAILLM

    source = inspect.getsource(OpenVINOGenAILLM)
    assert "def write(" in source
    assert "StreamingStatus" in source
    # Ensure old deprecated 'put' method is not present
    assert "def put(" not in source
