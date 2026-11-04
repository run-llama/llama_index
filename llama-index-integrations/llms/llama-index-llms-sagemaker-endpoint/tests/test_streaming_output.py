import json
from unittest.mock import Mock, patch

import pytest

from llama_index.llms.sagemaker_endpoint import SageMakerLLM
from llama_index.llms.sagemaker_endpoint.utils import IOHandler


@pytest.mark.parametrize(
    "text",
    [
        "the answer is 42",
        "great and neat",
        "data engineering",
        "generated_text",
        "answer]",
        "answer}",
        'answer"',
        '"quoted"',
        "line one\nline two",
        "path\\to\\file",
        "café 日本語",
        "",
    ],
)
def test_deserialize_streaming_output_preserves_text(text: str) -> None:
    payload = json.dumps([{"generated_text": text}], separators=(",", ":")).encode()
    assert IOHandler().deserialize_streaming_output(payload) == text


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b'[{"generated_text":"the ', "the "),
        (b"data engineering", "data engineering"),
        (b'great and neat"}]', "great and neat"),
        (b"]", "]"),
        (b"}", "}"),
        (b'\\"', '"'),
        (b'"}]', ""),
    ],
)
def test_deserialize_streaming_output_preserves_fragments(
    payload: bytes, expected: str
) -> None:
    assert IOHandler().deserialize_streaming_output(payload) == expected


def test_stream_complete_preserves_text_and_deltas() -> None:
    client = Mock()
    client.invoke_endpoint_with_response_stream.return_value = {
        "Body": [
            {"PayloadPart": {"Bytes": payload}}
            for payload in [
                b'[{"generated_text":"Question: ',
                b"the data",
                b' ends here]"}]',
            ]
        ]
    }
    with patch(
        "llama_index.llms.sagemaker_endpoint.base.get_aws_service_client",
        return_value=client,
    ):
        llm = SageMakerLLM(endpoint_name="synthetic-endpoint")
        responses = list(llm.stream_complete("Question: ", formatted=True))

    assert [response.delta for response in responses] == [
        "",
        "the data",
        " ends here]",
    ]
    assert [response.text for response in responses] == [
        "",
        "the data",
        "the data ends here]",
    ]
    client.invoke_endpoint_with_response_stream.assert_called_once()
