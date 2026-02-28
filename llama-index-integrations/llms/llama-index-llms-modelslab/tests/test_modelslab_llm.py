"""
Tests for llama-index-llms-modelslab.

Run with: pytest tests/ -v
"""

import os
from unittest.mock import patch

import pytest

from llama_index.llms.modelslab import ModelsLabLLM
from llama_index.llms.modelslab.base import MODELSLAB_API_BASE


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def llm():
    return ModelsLabLLM(
        model="llama-3.1-8b-uncensored",
        api_key="test-key-abc",
    )


# ── Constructor ───────────────────────────────────────────────────────────────


class TestModelsLabLLMConstructor:
    def test_default_model(self, llm):
        assert llm.model == "llama-3.1-8b-uncensored"

    def test_custom_model(self):
        llm = ModelsLabLLM(model="llama-3.1-70b-uncensored", api_key="k")
        assert llm.model == "llama-3.1-70b-uncensored"

    def test_default_api_base(self, llm):
        assert llm.api_base == MODELSLAB_API_BASE

    def test_default_api_base_value(self):
        assert MODELSLAB_API_BASE == "https://modelslab.com/uncensored-chat/v1"

    def test_is_chat_model_true(self, llm):
        assert llm.is_chat_model is True

    def test_context_window_128k(self, llm):
        assert llm.context_window == 131072

    def test_api_key_set(self, llm):
        assert llm.api_key == "test-key-abc"

    @patch.dict(os.environ, {"MODELSLAB_API_KEY": "env-key-xyz"})
    def test_reads_api_key_from_env(self):
        llm = ModelsLabLLM()
        assert llm.api_key == "env-key-xyz"

    def test_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MODELSLAB_API_KEY", None)
            with pytest.raises(ValueError, match="MODELSLAB_API_KEY"):
                ModelsLabLLM()

    def test_custom_api_base(self):
        custom = "https://custom.modelslab.com/v1"
        llm = ModelsLabLLM(api_key="k", api_base=custom)
        assert llm.api_base == custom

    def test_class_name(self, llm):
        assert llm.class_name() == "ModelsLabLLM"

    def test_import_from_package(self):
        from llama_index.llms.modelslab import ModelsLabLLM as ML
        from llama_index.llms.modelslab.base import ModelsLabLLM as MLBase

        assert ML is MLBase

    def test_inherits_from_openai_like(self, llm):
        from llama_index.llms.openai_like import OpenAILike

        assert isinstance(llm, OpenAILike)

    @patch.dict(os.environ, {"MODELSLAB_API_KEY": "env-key"})
    def test_explicit_key_overrides_env(self):
        llm = ModelsLabLLM(api_key="explicit-key")
        assert llm.api_key == "explicit-key"
