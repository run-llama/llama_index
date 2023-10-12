from unittest.mock import MagicMock, patch

import pytest
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
