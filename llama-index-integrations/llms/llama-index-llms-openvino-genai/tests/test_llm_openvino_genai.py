from llama_index.llms.openvino_genai import OpenVINOGenAILLM
from llama_index.core.base.llms.base import BaseLLM
import huggingface_hub as hf_hub


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in OpenVINOGenAILLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_streaming_completion():
    model_id = "OpenVINO/qwen3-0.6b-int4-ov"
    model_path = "qwen3-0.6b-int4-ov"
    hf_hub.snapshot_download(model_id, local_dir=model_path)
    ov_llm = OpenVINOGenAILLM(
        model_path=model_path,
        device="CPU",
    )
    ov_llm.config.max_new_tokens = 100
    response = ov_llm.stream_complete("Who is Paul Graham?")
    intermediate_response = None
    for chunk in response:
        intermediate_response = chunk

    assert len(intermediate_response.text) > 0


def test_streamer_write_accepts_single_token(monkeypatch):
    """write() must accept a single int token id, not only a list.

    Regression test: the non-list branch appended an undefined name and
    raised NameError on every single-token write.
    """
    from unittest.mock import MagicMock

    import openvino_genai

    monkeypatch.setattr(openvino_genai, "LLMPipeline", MagicMock())
    ov_llm = OpenVINOGenAILLM(model_path="unused")
    streamer = ov_llm._streamer
    streamer.write(1)
    assert streamer.tokens_cache == [1]
    streamer.write([2, 3])
    assert streamer.tokens_cache == [1, 2, 3]
