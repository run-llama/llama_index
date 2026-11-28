import pytest

from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.sagemaker_endpoint import SageMakerLLM
from llama_index.llms.sagemaker_endpoint.utils import IOHandler


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in SageMakerLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b'[{"generated_text":"hello world"}]', "hello world"),
        (
            b'[{"generated_text":"generated data extraction"}]',
            "generated data extraction",
        ),
        (b'[{"generated_text":"anaconda"}]', "anaconda"),
        (b'[{"generated_text":"The result is [1, 2]"}]', "The result is [1, 2]"),
    ],
)
def test_deserialize_streaming_output_preserves_text(raw, expected):
    """Affixes must be removed exactly, not as character sets."""
    assert IOHandler().deserialize_streaming_output(raw) == expected
