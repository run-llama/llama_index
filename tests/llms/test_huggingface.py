from unittest.mock import MagicMock, patch

import pytest
from llama_index.llms import ChatMessage, MessageRole
from llama_index.llms.huggingface import HuggingFaceInferenceAPI

STUB_MODEL_NAME = "placeholder_model"


@pytest.fixture(name="hf_inference_api")
def fixture_hf_inference_api() -> HuggingFaceInferenceAPI:
    with patch.dict("sys.modules", huggingface_hub=MagicMock()):
        return HuggingFaceInferenceAPI(model_name=STUB_MODEL_NAME)


class TestHuggingFaceInferenceAPI:
    def test_class_name(self, hf_inference_api: HuggingFaceInferenceAPI) -> None:
        assert HuggingFaceInferenceAPI.class_name() == HuggingFaceInferenceAPI.__name__
        assert hf_inference_api.class_name() == HuggingFaceInferenceAPI.__name__

    def test_instantiation(self) -> None:
        mock_hub = MagicMock()
        with patch.dict("sys.modules", huggingface_hub=mock_hub):
            llm = HuggingFaceInferenceAPI(model_name=STUB_MODEL_NAME)

        assert llm.model_name == STUB_MODEL_NAME

        # Confirm Clients are instantiated correctly
        mock_hub.InferenceClient.assert_called_once_with(
            model=STUB_MODEL_NAME, token=None, timeout=None, headers=None, cookies=None
        )
        mock_hub.AsyncInferenceClient.assert_called_once_with(
            model=STUB_MODEL_NAME, token=None, timeout=None, headers=None, cookies=None
        )

    def test_chat(self, hf_inference_api: HuggingFaceInferenceAPI) -> None:
        messages = [
            ChatMessage(content="Which movie is the best?"),
            ChatMessage(content="It's Die Hard for sure.", role=MessageRole.ASSISTANT),
            ChatMessage(content="Can you explain why?"),
        ]
        generated_response = (
            " It's based on the book of the same name by James Fenimore Cooper."
        )
        conversational_return = {
            "generated_text": generated_response,
            "conversation": {
                "generated_responses": ["It's Die Hard for sure.", generated_response],
                "past_user_inputs": [
                    "Which movie is the best?",
                    "Can you explain why?",
                ],
            },
        }

        with patch.object(
            hf_inference_api._sync_client,
            "conversational",
            return_value=conversational_return,
        ) as mock_conversational:
            response = hf_inference_api.chat(messages=messages)
        assert response.message.role == MessageRole.ASSISTANT
        assert response.message.content == generated_response
        mock_conversational.assert_called_once_with(
            text="Can you explain why?",
            past_user_inputs=["Which movie is the best?"],
            generated_responses=["It's Die Hard for sure."],
        )
