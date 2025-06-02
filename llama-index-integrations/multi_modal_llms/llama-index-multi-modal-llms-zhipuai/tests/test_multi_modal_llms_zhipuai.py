import base64
import os
import pytest
from unittest import mock
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.multi_modal_llms.base import MultiModalLLM
from llama_index.multi_modal_llms.zhipuai import ZhipuAIMultiModal
from zhipuai.types.chat.chat_completion import (
    Completion,
    CompletionChoice,
    CompletionMessage,
    CompletionUsage,
)


_FAKE_API_KEY = "ZHIPUAI_API_KEY"
_FAKE_CHAT_COMPLETIONS_RESPONSE = Completion(
    id="some_id",
    choices=[
        CompletionChoice(
            index=0,
            finish_reason="stop",
            message=CompletionMessage(
                role=MessageRole.ASSISTANT,
                content="nothing in the video",
            ),
        )
    ],
    usage=CompletionUsage(
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
    ),
)


def test_multi_modal_llm_class():
    names_of_base_classes = [b.__name__ for b in ZhipuAIMultiModal.__mro__]
    assert MultiModalLLM.__name__ in names_of_base_classes


def test_multi_modal_llm_series():
    llm = ZhipuAIMultiModal(model="glm-4v-plus", api_key="")
    assert llm.has_completions_api() is True
    llm = ZhipuAIMultiModal(model="cogview-3-plus", api_key="")
    assert llm.has_completions_api() is False
    llm = ZhipuAIMultiModal(model="cogvideox", api_key="")
    assert llm.has_videos_generations_api() is True


def test_get_glm_model_context_size():
    llm = ZhipuAIMultiModal(model="glm-4v", api_key="")
    assert llm.metadata.context_window > 0
    assert llm.model_kwargs
    with pytest.raises(ValueError):
        llm = ZhipuAIMultiModal(model="glm-x", api_key="")
        assert llm.metadata.context_window


def test_fake_llm_chat_and_complete():
    messages = [ChatMessage(role=MessageRole.USER, content="describe the video")]
    expected_response = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content="nothing in the video",
            additional_kwargs={"tool_calls": None},
        ),
        raw=_FAKE_CHAT_COMPLETIONS_RESPONSE,
    )
    llm = ZhipuAIMultiModal(model="glm-4v-plus", api_key=_FAKE_API_KEY)

    with mock.patch.object(
        llm._client.chat.completions,
        "create",
        return_value=_FAKE_CHAT_COMPLETIONS_RESPONSE,
    ):
        actual_response = llm.chat(messages=messages)
        assert actual_response == expected_response


@pytest.mark.skipif(
    os.getenv("ZHIPUAI_API_KEY") is None, reason="ZHIPUAI_API_KEY not set"
)
def test_llm_chat_and_complete():
    # test glm-4v completion
    llm = ZhipuAIMultiModal(model="glm-4v", api_key=os.getenv("ZHIPUAI_API_KEY"))
    with open(os.getenv("ZHIPUAI_TEST_VIDEO"), "rb") as video_file:
        video_base = base64.b64encode(video_file.read()).decode("utf-8")
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content=[
                {"type": "video_url", "video_url": {"url": video_base}},
                {"type": "text", "text": "descript the video"},
            ],
        ),
    ]
    assert llm.chat(messages)
    assert llm.complete("descript the video", video_url=video_base)
    assert llm.stream_chat(messages)
    assert llm.stream_complete("descript the video", video_url=video_base)
    # test cogview or cogvideox
    llm = ZhipuAIMultiModal(
        model="cogvideox", api_key=os.getenv("ZHIPUAI_API_KEY"), size="768x1344"
    )
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content=[{"type": "text", "text": "a bird flying in the sky"}],
        ),
    ]
    assert llm.chat(messages)
    assert llm.complete("a bird flying in the sky")


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("ZHIPUAI_API_KEY") is None, reason="ZHIPUAI_API_KEY not set"
)
async def test_async_llm_chat_and_complete():
    # test glm-4v completion
    llm = ZhipuAIMultiModal(model="glm-4v", api_key=os.getenv("ZHIPUAI_API_KEY"))
    with open(os.getenv("ZHIPUAI_TEST_VIDEO"), "rb") as video_file:
        video_base = base64.b64encode(video_file.read()).decode("utf-8")
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content=[
                {"type": "video_url", "video_url": {"url": video_base}},
                {"type": "text", "text": "descript the video"},
            ],
        ),
    ]
    assert await llm.astream_chat(messages)
    assert await llm.astream_complete("descript the video", video_url=video_base)
    llm = ZhipuAIMultiModal(
        model="cogview-3-plus", api_key=os.getenv("ZHIPUAI_API_KEY")
    )
    assert await llm.acomplete("draw a bird flying in the sky")
