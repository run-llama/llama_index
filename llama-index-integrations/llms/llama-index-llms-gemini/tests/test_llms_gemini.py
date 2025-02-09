import os

from llama_index.core.tools.function_tool import FunctionTool
import pytest
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatMessage, ImageBlock, MessageRole
from llama_index.llms.gemini import Gemini
from llama_index.llms.gemini.utils import chat_message_to_gemini

from google.ai.generativelanguage_v1beta.types import (
    FunctionCallingConfig,
    ToolConfig,
)


def test_embedding_class() -> None:
    names_of_base_classes = [b.__name__ for b in Gemini.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_chat_message_to_gemini() -> None:
    msg = ChatMessage("Some content")
    assert chat_message_to_gemini(msg) == {
        "role": MessageRole.USER,
        "parts": ["Some content"],
    }

    msg = ChatMessage("Some content")
    msg.blocks.append(ImageBlock(image=b"foo", image_mimetype="image/png"))
    assert chat_message_to_gemini(msg) == {
        "role": MessageRole.USER,
        "parts": [{"text": "Some content"}, {"data": b"foo", "mime_type": "image/png"}],
    }


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_generate_image_prompt() -> None:
    msg = ChatMessage("Tell me the brand of the car in this image:")
    msg.blocks.append(
        ImageBlock(
            url="https://upload.wikimedia.org/wikipedia/commons/5/52/Ferrari_SP_FFX.jpg"
        )
    )
    response = Gemini(model="models/gemini-1.5-flash").chat(messages=[msg])
    assert "ferrari" in str(response).lower()


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_chat_stream() -> None:
    msg = ChatMessage("List three types of software testing strategies")
    response = list(Gemini(model="models/gemini-1.5-flash").stream_chat(messages=[msg]))
    assert response


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None, reason="GOOGLE_API_KEY not set"
)
def test_chat_with_tools() -> None:
    def add(a: int, b: int) -> int:
        """Add two integers and returns the result integer."""
        return a + b

    add_tool = FunctionTool.from_defaults(fn=add)
    msg = ChatMessage("What is the result of adding 2 and 3?")
    model = Gemini(model="models/gemini-1.5-flash")
    response = model.chat_with_tools(
        user_msg=msg,
        tools=[add_tool],
        tool_config=ToolConfig(
            function_calling_config=FunctionCallingConfig(
                mode=FunctionCallingConfig.Mode.ANY
            )
        ),
    )

    tool_calls = model.get_tool_calls_from_response(response)
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "add"
    assert tool_calls[0].tool_kwargs == {"a": 2, "b": 3}

    assert len(response.additional_kwargs["tool_calls"]) >= 1
