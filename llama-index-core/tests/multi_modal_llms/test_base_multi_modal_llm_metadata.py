"""Unit tests for `llama_index.core.multi_modal_llms.MultiModalLLMMetadata`."""

from llama_index.core.multi_modal_llms import MultiModalLLMMetadata


class TestMultiModalLLMMetadata:
    def test_default_values(self):
        metadata = MultiModalLLMMetadata()
        assert metadata.model_name == "unknown"
        assert metadata.is_chat_model is False
        assert metadata.is_function_calling_model is False
        assert metadata.context_window is not None
        assert metadata.num_output is not None
        assert metadata.num_input_files is not None

    def test_custom_values(self):
        metadata = MultiModalLLMMetadata(
            model_name="test-model",
            context_window=2048,
            num_output=512,
            num_input_files=5,
            is_function_calling_model=True,
            is_chat_model=True,
        )
        assert metadata.model_name == "test-model"
        assert metadata.context_window == 2048
        assert metadata.num_output == 512
        assert metadata.num_input_files == 5
        assert metadata.is_function_calling_model is True
        assert metadata.is_chat_model is True
