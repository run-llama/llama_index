import json

from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.sagemaker_endpoint import SageMakerLLM
from llama_index.llms.sagemaker_endpoint.utils import IOHandler


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in SageMakerLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_deserialize_streaming_output_preserves_generated_text():
    handler = IOHandler()
    generated_text = 'the data answer ends with "'
    response = json.dumps(
        [{"generated_text": generated_text}],
        separators=(",", ":"),
    ).encode("utf-8")

    assert handler.deserialize_streaming_output(response) == generated_text
