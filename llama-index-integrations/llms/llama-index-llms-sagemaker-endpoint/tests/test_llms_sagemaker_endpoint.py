import pytest

from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.sagemaker_endpoint import SageMakerLLM
from llama_index.llms.sagemaker_endpoint.utils import IOHandler


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in SageMakerLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        # Regression: lstrip/rstrip treated the envelope as a character set,
        # so generated text starting with any of its characters (t, g, d, a,
        # e, n, r, x, _, etc.) or ending with '"', '}' or ']' was corrupted.
        (b'[{"generated_text":"the answer is 42"}]', "the answer is 42"),
        (b'[{"generated_text":"great and neat"}]', "great and neat"),
        (b'[{"generated_text":"data engineering"}]', "data engineering"),
        (b'[{"generated_text":"result: {}"}]', "result: {}"),
        (b'[{"generated_text":"Hello world"}]', "Hello world"),
        # Escaped quotes inside the generated text are preserved by json parsing.
        (b'[{"generated_text":"he said \\"hi\\""}]', 'he said "hi"'),
    ],
)
def test_deserialize_streaming_output_preserves_text(payload, expected):
    handler = IOHandler()
    assert handler.deserialize_streaming_output(payload) == expected
