import os
import pytest
from typing import Any
from pytest_httpx import HTTPXMock

from llama_index.llms.heroku import Heroku


@pytest.fixture()
def mock_heroku_models(httpx_mock: HTTPXMock):
    """Mock Heroku models endpoint response."""
    mock_response = {
        "data": [
            {
                "id": "claude-3-5-haiku",
                "object": "model",
                "created": 1234567890,
                "owned_by": "heroku",
                "root": "claude-3-5-haiku",
            }
        ]
    }

    httpx_mock.add_response(
        url="https://test-app.herokuapp.com/v1/models",
        method="GET",
        json=mock_response,
        status_code=200,
    )


def get_api_key(instance: Any) -> str:
    """Helper function to get API key from instance."""
    return instance.api_key


def test_create_without_api_key_raises_error() -> None:
    """Test that creating without API key raises ValueError."""
    with pytest.raises(ValueError, match="API key is required"):
        Heroku()


def test_create_without_inference_url_raises_error() -> None:
    """Test that creating without inference URL raises ValueError."""
    with pytest.raises(ValueError, match="Inference URL is required"):
        Heroku(api_key="test-key")


def test_create_without_model_raises_error() -> None:
    """Test that creating without model raises ValueError."""
    with pytest.raises(ValueError, match="Model is required"):
        Heroku(api_key="test-key", inference_url="https://test-app.herokuapp.com")


def test_create_with_all_parameters() -> None:
    """Test creating with all required parameters."""
    instance = Heroku(
        model="claude-3-5-haiku",
        api_key="test-key",
        inference_url="https://test-app.herokuapp.com",
    )
    assert instance.api_key == "test-key"
    assert instance.api_base == "https://test-app.herokuapp.com/v1"
    assert instance.model == "claude-3-5-haiku"


def test_api_key_from_environment() -> None:
    """Test that API key is read from environment variable."""
    try:
        os.environ["INFERENCE_KEY"] = "env-key"
        os.environ["INFERENCE_URL"] = "https://test-app.herokuapp.com"
        os.environ["INFERENCE_MODEL_ID"] = "claude-3-5-haiku"

        instance = Heroku()
        assert instance.api_key == "env-key"
        assert instance.api_base == "https://test-app.herokuapp.com/v1"
        assert instance.model == "claude-3-5-haiku"
    finally:
        # Clean up environment variables
        for key in ["INFERENCE_KEY", "INFERENCE_URL", "INFERENCE_MODEL_ID"]:
            if key in os.environ:
                del os.environ[key]


def test_parameter_overrides_environment() -> None:
    """Test that parameters override environment variables."""
    try:
        os.environ["INFERENCE_KEY"] = "env-key"
        os.environ["INFERENCE_URL"] = "https://env-app.herokuapp.com"
        os.environ["INFERENCE_MODEL_ID"] = "env-model"

        instance = Heroku(
            model="param-model",
            api_key="param-key",
            inference_url="https://param-app.herokuapp.com",
        )
        assert instance.api_key == "param-key"
        assert instance.api_base == "https://param-app.herokuapp.com/v1"
        assert instance.model == "param-model"
    finally:
        # Clean up environment variables
        for key in ["INFERENCE_KEY", "INFERENCE_URL", "INFERENCE_MODEL_ID"]:
            if key in os.environ:
                del os.environ[key]


def test_model_parameter_overrides_environment() -> None:
    """Test that model parameter overrides environment variable."""
    try:
        os.environ["INFERENCE_MODEL_ID"] = "env-model"
        instance = Heroku(
            model="explicit-model",
            api_key="test-key",
            inference_url="https://test-app.herokuapp.com",
        )
        assert instance.model == "explicit-model"
    finally:
        if "INFERENCE_MODEL_ID" in os.environ:
            del os.environ["INFERENCE_MODEL_ID"]


def test_model_from_environment() -> None:
    """Test that model is read from environment variable when not provided."""
    try:
        os.environ["INFERENCE_MODEL_ID"] = "env-model"
        instance = Heroku(
            api_key="test-key", inference_url="https://test-app.herokuapp.com"
        )
        assert instance.model == "env-model"
    finally:
        if "INFERENCE_MODEL_ID" in os.environ:
            del os.environ["INFERENCE_MODEL_ID"]


@pytest.mark.integration
def test_missing_api_key_error() -> None:
    """Test that missing API key results in proper error."""
    with pytest.raises(ValueError, match="API key is required"):
        Heroku(inference_url="https://test-app.herokuapp.com", model="test-model")


@pytest.mark.integration
def test_missing_inference_url_error() -> None:
    """Test that missing inference URL results in proper error."""
    with pytest.raises(ValueError, match="Inference URL is required"):
        Heroku(api_key="test-key", model="test-model")


@pytest.mark.integration
def test_missing_model_error() -> None:
    """Test that missing model results in proper error."""
    with pytest.raises(ValueError, match="Model is required"):
        Heroku(api_key="test-key", inference_url="https://test-app.herokuapp.com")
