###Test OCI Generative AI LLM service
from unittest.mock import MagicMock
from typing import Any

import pytest
from pytest import MonkeyPatch

from llama_index.llms.oci_genai import OCIGenAI
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole


class MockResponseDict(dict):
    def __getattr__(self, val) -> Any:  # type: ignore[no-untyped-def]
        return self[val]


@pytest.mark.parametrize(
    "test_model_id", ["cohere.command", "cohere.command-light", "meta.llama-2-70b-chat"]
)
def test_llm_complete(monkeypatch: MonkeyPatch, test_model_id: str) -> None:
    """Test valid completion call to OCI Generative AI LLM service."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)

    provider = llm._provider.__class__.__name__

    def mocked_response(*args):  # type: ignore[no-untyped-def]
        response_text = "This is the completion."

        if provider == "CohereProvider":
            return MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "inference_response": MockResponseDict(
                                {
                                    "generated_texts": [
                                        MockResponseDict(
                                            {
                                                "text": response_text,
                                            }
                                        )
                                    ]
                                }
                            )
                        }
                    ),
                }
            )
        elif provider == "MetaProvider":
            return MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "inference_response": MockResponseDict(
                                {
                                    "choices": [
                                        MockResponseDict(
                                            {
                                                "text": response_text,
                                            }
                                        )
                                    ]
                                }
                            )
                        }
                    ),
                }
            )
        else:
            return None

    monkeypatch.setattr(llm._client, "generate_text", mocked_response)

    output = llm.complete("This is a prompt.", temperature=0.2)
    assert output.text == "This is the completion."


@pytest.mark.parametrize(
    "test_model_id", ["cohere.command-r-16k", "meta.llama-3-70b-instruct"]
)
def test_llm_chat(monkeypatch: MonkeyPatch, test_model_id: str) -> None:
    """Test valid chat call to OCI Generative AI LLM service."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model=test_model_id, client=oci_gen_ai_client)

    provider = llm._provider.__class__.__name__

    def mocked_response(*args):  # type: ignore[no-untyped-def]
        response_text = "Assistant chat reply."
        response = None
        if provider == "CohereProvider":
            response = MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "text": response_text,
                                }
                            )
                        }
                    ),
                }
            )
        elif provider == "MetaProvider":
            response = MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "choices": [
                                        MockResponseDict(
                                            {
                                                "message": MockResponseDict(
                                                    {
                                                        "content": [
                                                            MockResponseDict(
                                                                {
                                                                    "text": response_text,
                                                                }
                                                            )
                                                        ]
                                                    }
                                                )
                                            }
                                        )
                                    ]
                                }
                            )
                        }
                    ),
                }
            )
        return response

    monkeypatch.setattr(llm._client, "chat", mocked_response)

    messages = [
        ChatMessage(role="user", content="User message"),
    ]

    expected = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT, content="Assistant chat reply."
        ),
        raw=llm._client.chat.__dict__,
    )

    actual = llm.chat(messages, temperature=0.2)
    assert actual == expected
