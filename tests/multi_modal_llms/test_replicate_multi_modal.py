from typing import Any

from llama_index.multi_modal_llms.replicate_multi_modal import ReplicateMultiModal
from llama_index.schema import ImageDocument
from pytest import MonkeyPatch


def mock_completion(*args: Any, **kwargs: Any) -> dict:
    # Example taken from https://replicate.com/
    return {
        "completed_at": "2023-11-03T17:37:40.927121Z",
        "created_at": "2023-11-03T17:36:22.310997Z",
        "id": "oieao3tbk6er3lj3a7woe3yyjq",
        "input": {
            "image": "https://replicate.delivery/pbxt/JfvBi04QfleIeJ3ASiBEMbJvhTQKWKLjKaajEbuhO1Y0wPHd/view.jpg",
            "top_p": 1,
            "prompt": "Are you allowed to swim here?",
            "max_tokens": 1024,
            "temperature": 0.2,
        },
        "metrics": {"predict_time": 4.837953},
        "output": [
            "Yes, ",
            "you ",
            "are ",
            "allowed ",
        ],
        "started_at": "2023-11-03T17:37:36.089168Z",
        "status": "succeeded",
        "urls": {
            "get": "https://api.replicate.com/v1/predictions/oieao3tbk6er3lj3a7woe3yyjq",
            "cancel": "https://api.replicate.com/v1/predictions/oieao3tbk6er3lj3a7woe3yyjq/cancel",
        },
        "version": "2facb4a474a0462c15041b78b1ad70952ea46b5ec6ad29583c0b29dbd4249591",
    }


def test_completion_model_basic(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llama_index.multi_modal_llms.ReplicateMultiModal.complete", mock_completion
    )

    llm = ReplicateMultiModal(model="llava")
    prompt = "test prompt"
    response = llm.complete(prompt, [ImageDocument()])
    assert "".join(response["output"]) == "Yes, you are allowed "
