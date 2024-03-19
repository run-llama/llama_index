"""Test OCI Generative AI LLM service"""
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch

from llama_index.llms.ocigenai import OCIGenAI
from llama_index.core.llms import ChatMessage


class MockResponseDict(dict):
    def __getattr__(self, val):  # type: ignore[no-untyped-def]
        return self[val]


@pytest.mark.requires("oci")
@pytest.mark.parametrize(
    "test_model_id", ["cohere.command", "cohere.command-light", "meta.llama-2-70b-chat"]
)
def test_llm_call(monkeypatch: MonkeyPatch, test_model_id: str) -> None:
    """Test valid call to OCI Generative AI LLM service."""
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

        if provider == "MetaProvider":
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

    monkeypatch.setattr(llm._client, "generate_text", mocked_response)

    output = llm.complete("This is a prompt.", temperature=0.2)
    assert output.text == "This is the completion."

    messages = [
        ChatMessage(role="user", content="This is a prompt."),
    ]
    
    output = llm.chat(messages, temperature=0.2)
    assert str(output.message) == "assistant: This is the completion."
