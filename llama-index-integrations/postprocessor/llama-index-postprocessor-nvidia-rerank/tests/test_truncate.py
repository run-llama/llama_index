from typing import Any, Literal, Optional

import pytest
import re
import respx
import json
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank
from llama_index.core.schema import NodeWithScore, Document


@pytest.fixture()
def mock_v1_models(respx_mock: respx.MockRouter) -> None:
    respx_mock.get("https://integrate.api.nvidia.com/v1/models").respond(
        json={
            "data": [
                {
                    "id": "mock-model",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "OWNER",
                }
            ]
        }
    )


@pytest.fixture()
def mock_v1_ranking(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(
        re.compile(r"https://ai\.api\.nvidia\.com/v1/.*/reranking")
    ).respond(
        json={
            "rankings": [
                {"index": 0, "logit": 4.2},
            ]
        }
    )


@pytest.fixture()
def mock(mock_v1_models: None, mock_v1_ranking: None) -> None:
    pass


@pytest.mark.parametrize(
    "truncate",
    [
        None,
        "END",
        "NONE",
    ],
)
def test_truncate_passed(
    mock: None,
    respx_mock: respx.MockRouter,
    truncate: Optional[Literal["END", "NONE"]],
) -> None:
    client = NVIDIARerank(
        api_key="BOGUS",
        **({"truncate": truncate} if truncate else {}),
    )
    response = client.postprocess_nodes(
        [NodeWithScore(node=Document(text="Nothing really."))],
        query_str="What is it?",
    )

    assert len(response) == 1

    assert len(respx.calls) > 0
    last_call = list(respx.calls)[-1]
    request_payload = json.loads(last_call.request.content.decode("utf-8"))
    if truncate is None:
        assert "truncate" not in request_payload
    else:
        assert "truncate" in request_payload
        assert request_payload["truncate"] == truncate


@pytest.mark.parametrize("truncate", [True, False, 1, 0, 1.0, "START", "BOGUS"])
def test_truncate_invalid(truncate: Any) -> None:
    with pytest.raises(ValueError):
        NVIDIARerank(truncate=truncate)


@pytest.mark.integration()
@pytest.mark.parametrize("truncate", ["END"])
def test_truncate_positive(model: str, mode: dict, truncate: str) -> None:
    query = "What is acceleration?"
    nodes = [
        NodeWithScore(node=Document(text="NVIDIA " * length))
        for length in [32, 1024, 64, 128, 2048, 256, 512]
    ]
    client = NVIDIARerank(model=model, top_n=len(nodes), truncate=truncate, **mode)
    response = client.postprocess_nodes(nodes, query_str=query)
    print(response)
    assert len(response) == len(nodes)


@pytest.mark.integration()
@pytest.mark.parametrize("truncate", [None, "NONE"])
def test_truncate_negative(model: str, mode: dict, truncate: str) -> None:
    if model == "nv-rerank-qa-mistral-4b:1":
        pytest.skip(
            "truncation is inconsistent across models, "
            "nv-rerank-qa-mistral-4b:1 truncates by default "
            "while others do not"
        )
    query = "What is acceleration?"
    nodes = [
        NodeWithScore(node=Document(text="NVIDIA " * length))
        for length in [32, 1024, 64, 128, 2048, 256, 512]
    ]
    client = NVIDIARerank(
        model=model, **mode, **({"truncate": truncate} if truncate else {})
    )
    with pytest.raises(Exception) as e:
        client.postprocess_nodes(nodes, query_str=query)
    assert "400" in str(e.value)
    # assert "exceeds maximum allowed" in str(e.value)
