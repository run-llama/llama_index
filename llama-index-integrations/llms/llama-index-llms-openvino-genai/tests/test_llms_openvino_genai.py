import pytest

try:
    import openvino_genai

    has_openvino_genai = True
except ImportError:
    has_openvino_genai = False


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
    import queue
    from typing import Union
    from unittest.mock import MagicMock

    tokenizer_mock = MagicMock()
    tokenizer_mock.decode = MagicMock(return_value="hello")

    # Import the module to access the nested classes through instantiation
    # We test through the ChunkStreamer since it's what gets used
    from llama_index.llms.openvino_genai.base import OpenVINOGenAILLM

    # Access the class source to verify write method exists
    import inspect

    source = inspect.getsource(OpenVINOGenAILLM)
    assert "def write(self, token" in source
    assert "StreamingStatus" in source
    # Ensure old deprecated 'put' method is not present
    assert "def put(self, token" not in source
