from typing import Any, Sequence

from llama_index.llms.anyscale import Anyscale
from llama_index.llms.base import ChatMessage, ChatResponse, LLMMetadata, MessageRole


class MockAnyscale(Anyscale):
    def __init__(self) -> None:
        super().__init__(model="MOCK_MODEL", api_key="MOCK_KEY", callback_manager=None)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata()

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content="Test Chat Response"
            ),
            raw=None,
        )


def test_basic() -> None:
    llm = MockAnyscale()
    test_prompt = "test prompt"
    response = llm.complete(test_prompt)
    assert len(response.text) > 0

    message = ChatMessage(role="user", content=test_prompt)
    chat_response = llm.chat([message])
    assert chat_response.message.content is not None
    assert len(chat_response.message.content) > 0
