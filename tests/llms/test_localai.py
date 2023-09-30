from unittest.mock import patch

from llama_index.llms import LocalAI


def test_interfaces() -> None:
    llm = LocalAI(model="placeholder")
    assert llm.class_name() == type(llm).__name__
    assert llm.model == "placeholder"


def test_completion() -> None:
    llm = LocalAI(model="models/llama-2-13b-ensemble-v5.Q4_K_M.gguf")

    text = "...\n\nIt was just another day at the office. The sun had ris"
    with patch(
        "llama_index.llms.openai.completion_with_retry",
        return_value={
            "id": "123",
            "object": "text_completion",
            "created": 1696036786,
            "model": "models/llama-2-13b-ensemble-v5.Q4_K_M.gguf",
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 13, "completion_tokens": 16, "total_tokens": 29},
        },
    ) as mock_completion:
        response = llm.complete("A long time ago in a galaxy far, far away")
    assert response.text == text
    mock_completion.assert_called_once()
    # Check we remove the max_tokens if unspecified
    assert "max_tokens" not in mock_completion.call_args.kwargs
