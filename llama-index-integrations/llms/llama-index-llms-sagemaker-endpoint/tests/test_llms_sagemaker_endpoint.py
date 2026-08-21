from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.sagemaker_endpoint import SageMakerLLM
from llama_index.llms.sagemaker_endpoint.utils import IOHandler


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in SageMakerLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_deserialize_streaming_output_preserves_text():
    # Regression test: lstrip/rstrip were used as prefix/suffix removal, which
    # ate leading/trailing characters from the generated text itself.
    handler = IOHandler()
    assert (
        handler.deserialize_streaming_output(b'[{"generated_text":"the answer is 42"}]')
        == "the answer is 42"
    )
    assert (
        handler.deserialize_streaming_output(b'[{"generated_text":"great and neat"}]')
        == "great and neat"
    )
    assert (
        handler.deserialize_streaming_output(b'[{"generated_text":"data engineering"}]')
        == "data engineering"
    )


def test_deserialize_streaming_output_handles_escaped_quotes():
    handler = IOHandler()
    assert (
        handler.deserialize_streaming_output(b'[{"generated_text":"he said \\"hi\\""}]')
        == 'he said "hi"'
    )
