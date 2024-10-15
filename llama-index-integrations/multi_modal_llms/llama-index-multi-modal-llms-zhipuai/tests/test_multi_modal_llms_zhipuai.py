import base64
import os
import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.multi_modal_llms.base import MultiModalLLM
from llama_index.multi_modal_llms.zhipuai import ZhipuAIMultiModal


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


@pytest.mark.asyncio()
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
