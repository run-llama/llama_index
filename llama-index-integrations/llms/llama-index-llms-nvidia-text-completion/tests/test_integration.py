import pytest
from llama_index.core.base.llms.types import (
    CompletionResponse,
)
from llama_index.llms.nvidia_text_completion import NVIDIATextCompletion
from llama_index.llms.nvidia_text_completion.utils import COMPLETION_MODEL_TABLE


@pytest.mark.parametrize("model", COMPLETION_MODEL_TABLE)
@pytest.mark.integration()
def test_complete(model: str) -> None:
    response = NVIDIATextCompletion(model=model).complete("Hello")
    assert isinstance(response, CompletionResponse)
    assert isinstance(response.text, str)


@pytest.mark.parametrize("model", COMPLETION_MODEL_TABLE)
@pytest.mark.integration()
@pytest.mark.asyncio()
async def test_acomplete(model: str) -> None:
    response = await NVIDIATextCompletion(model=model).acomplete("Hello")
    assert isinstance(response, CompletionResponse)
    assert isinstance(response.text, str)
