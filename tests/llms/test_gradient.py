"""Test GradientAI."""

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from llama_index.llms.gradient import GradientBaseModelLLM, GradientModelAdapterLLM
from llama_index.llms.types import CompletionResponse


class GradientModel(MagicMock):
    """MockGradientModel."""

    def complete(self, query: str, max_generated_token_count: int) -> Any:
        """Just duplicate the query m times."""
        output = MagicMock()
        output.generated_output = f"{query*max_generated_token_count}"
        return output

    async def acomplete(self, query: str, max_generated_token_count: int) -> Any:
        """Just duplicate the query m times."""
        output = MagicMock()
        output.generated_output = f"{query*max_generated_token_count}"
        return output


class MockGradient(MagicMock):
    """Mock Gradient package."""

    def get_base_model(self, base_model_slug: str) -> GradientModel:
        assert base_model_slug == "dummy-base-model"

        return GradientModel()

    def close(self) -> None:
        """Mock Gradient completion."""
        return

    def get_model_adapter(self, model_adapter_id: str) -> GradientModel:
        assert model_adapter_id == "dummy-adapter-model"
        return GradientModel()


class MockGradientaiPackage(MagicMock):
    """Mock Gradientai package."""

    Gradient = MockGradient


def test_gradient_base() -> None:
    """Test Gradient."""
    # Set up fake package here
    with patch.dict(sys.modules, {"gradientai": MockGradientaiPackage()}):
        n_tokens = 2
        gradientllm = GradientBaseModelLLM(
            access_token="dummy-token",
            base_model_slug="dummy-base-model",
            workspace_id="dummy-workspace",
            max_tokens=n_tokens,
        )
        response = gradientllm.complete("hello world")
        assert isinstance(response, CompletionResponse)
        assert response.text == "hello world" * n_tokens


def test_gradient_adapter() -> None:
    # Set up fake package here
    with patch.dict(sys.modules, {"gradientai": MockGradientaiPackage()}):
        n_tokens = 5
        gradientllm = GradientModelAdapterLLM(
            access_token="dummy-token",
            model_adapter_id="dummy-adapter-model",
            workspace_id="dummy-workspace",
            max_tokens=n_tokens,
        )
        response = gradientllm.complete("hello world")
        assert isinstance(response, CompletionResponse)
        assert response.text == "hello world" * n_tokens


@pytest.mark.asyncio()
async def test_async_gradient_Base() -> None:
    """Test Gradient."""
    # Set up fake package here, uses the same package.
    with patch.dict(sys.modules, {"gradientai": MockGradientaiPackage()}):
        n_tokens = 3
        gradientllm = GradientBaseModelLLM(
            access_token="dummy-token",
            base_model_slug="dummy-base-model",
            workspace_id="dummy-workspace",
            max_tokens=n_tokens,
        )
        response = await gradientllm.acomplete("hello world")
        assert isinstance(response, CompletionResponse)
        assert response.text == "hello world" * n_tokens


@pytest.mark.asyncio()
async def test_async_gradient_adapter() -> None:
    with patch.dict(sys.modules, {"gradientai": MockGradientaiPackage()}):
        sys.modules["gradientai"] = MockGradientaiPackage()
        n_tokens = 4
        gradientllm = GradientModelAdapterLLM(
            access_token="dummy-token",
            model_adapter_id="dummy-adapter-model",
            workspace_id="dummy-workspace",
            max_tokens=n_tokens,
        )
        response = await gradientllm.acomplete("hello world")
        assert isinstance(response, CompletionResponse)
        assert response.text == "hello world" * n_tokens
