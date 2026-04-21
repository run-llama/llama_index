from types import SimpleNamespace

from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in LlamaCPP.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_completion_to_prompt_v3_instruct():
    output = (
        "<|start_header_id|>system<|end_header_id|>\n\nSYSTEM PROMPT<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\nUSER MESSAGE<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    assert completion_to_prompt_v3_instruct("USER MESSAGE", "SYSTEM PROMPT") == output


def test_messages_to_prompt_v3_instruct():
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="The next sentence said by the assistant is true.",
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="The previous sentence said by the user is false.",
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="Do you think the last sentence spoke by me is true or false?",
        ),
    ]
    output = (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "SYSTEM PROMPT<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "The next sentence said by the assistant is true.<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "The previous sentence said by the user is false.<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Do you think the last sentence spoke by me is true or false?<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    assert messages_to_prompt_v3_instruct(messages, "SYSTEM PROMPT") == output


def test_context_window_uses_model_default_when_zero(monkeypatch, tmp_path):
    import llama_index.llms.llama_cpp.base as llama_cpp_base

    class FakeLlama:
        last_kwargs = None

        def __init__(self, model_path: str, **kwargs):
            FakeLlama.last_kwargs = kwargs
            self.context_params = SimpleNamespace(n_ctx=8192)

        def n_ctx(self) -> int:
            return 8192

    monkeypatch.setattr(llama_cpp_base, "Llama", FakeLlama)

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"")

    llm = LlamaCPP(model_path=str(model_path), context_window=0)
    assert FakeLlama.last_kwargs["n_ctx"] == 0
    assert llm.context_window == 8192
