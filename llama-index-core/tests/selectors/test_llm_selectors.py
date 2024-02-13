from unittest.mock import patch

from llama_index.core.llms import CompletionResponse
from llama_index.core.selectors.llm_selectors import (
    LLMMultiSelector,
    LLMSingleSelector,
)
from llama_index.core.service_context import ServiceContext
from tests.mock_utils.mock_predict import _mock_single_select


def test_llm_single_selector() -> None:
    service_context = ServiceContext.from_defaults(llm=None, embed_model=None)
    selector = LLMSingleSelector.from_defaults(service_context=service_context)

    with patch.object(
        type(service_context.llm),
        "complete",
        return_value=CompletionResponse(text=_mock_single_select()),
    ) as mock_complete:
        result = selector.select(
            choices=["apple", "pear", "peach"], query="what is the best fruit?"
        )
    assert result.ind == 0
    mock_complete.assert_called_once()
    assert mock_complete.call_args.args[0].count("Here is an example") <= 1


def test_llm_multi_selector(
    mock_service_context: ServiceContext,
) -> None:
    selector = LLMMultiSelector.from_defaults(service_context=mock_service_context)

    choices = [
        "apple",
        "pear",
        "peach",
    ]
    query = "what is the best fruit?"

    result = selector.select(choices, query)
    assert result.inds == [0, 1, 2]


def test_llm_multi_selector_max_choices(
    mock_service_context: ServiceContext,
) -> None:
    selector = LLMMultiSelector.from_defaults(
        service_context=mock_service_context, max_outputs=2
    )

    choices = [
        "apple",
        "pear",
        "peach",
    ]
    query = "what is the best fruit?"

    result = selector.select(choices, query)
    assert result.inds == [0, 1]
