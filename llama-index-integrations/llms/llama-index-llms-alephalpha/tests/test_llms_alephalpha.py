from unittest.mock import MagicMock, patch

from llama_index.llms.alephalpha.base import AlephAlpha
from llama_index.llms.alephalpha.utils import extract_additional_info_from_response


def test_alephalpha_instantiation():
    model = "luminous-base"
    token = "test_token"
    aleph_alpha_instance = AlephAlpha(model=model, token=token)

    assert aleph_alpha_instance.model == model
    assert aleph_alpha_instance.token == token
    assert (
        aleph_alpha_instance.temperature == AlephAlpha.__fields__["temperature"].default
    )


def test_complete_method():
    mock_completion = MagicMock()
    mock_completion.completion = "Test completion"

    mock_response_json = {
        "completion": "Test completion",
    }

    mock_response = MagicMock()
    mock_response.completions = [mock_completion]
    mock_response.to_json.return_value = mock_response_json

    with patch(
        "llama_index.llms.alephalpha.base.AlephAlpha._get_client"
    ) as mock_get_client:
        mock_client = MagicMock()
        mock_client.complete.return_value = mock_response
        mock_get_client.return_value = mock_client

        aleph_alpha_instance = AlephAlpha()
        response = aleph_alpha_instance.complete("Test prompt")

        assert response.text == "Test completion"
        assert response.raw == mock_response_json


def test_extract_additional_info_from_response():
    mock_completion = {
        "log_probs": [{"token": "test", "log_prob": -0.5}],
        "raw_completion": "Raw test completion",
        "finish_reason": "length",
    }

    mock_response = MagicMock()
    mock_response.model_version = "luminous-base-control"
    mock_response.completions = [MagicMock(**mock_completion)]
    mock_response.json.return_value = {
        "model_version": "luminous-base-control",
        "completions": [mock_completion],
    }

    extracted_info = extract_additional_info_from_response(mock_response)

    assert extracted_info["model_version"] == "luminous-base-control"
    assert extracted_info["log_probs"] == [{"token": "test", "log_prob": -0.5}]
    assert extracted_info["raw_completion"] == "Raw test completion"
    assert extracted_info["finish_reason"] == "length"
