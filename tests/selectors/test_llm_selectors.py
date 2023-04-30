from gpt_index.indices.service_context import ServiceContext
from gpt_index.selectors.llm_selectors import LLMMultiSelector, LLMSingleSelector


def test_llm_single_selector(
    mock_service_context: ServiceContext,
):
    selector = LLMSingleSelector.from_defaults(service_context=mock_service_context)

    choices = [
        "apple",
        "pear",
        "peach",
    ]
    query = "what is the best fruit?"

    result = selector.select(choices, query)
    assert result.selection_ind == 1


def test_llm_multi_selector(
    mock_service_context: ServiceContext,
):
    selector = LLMMultiSelector.from_defaults(service_context=mock_service_context)

    choices = [
        "apple",
        "pear",
        "peach",
    ]
    query = "what is the best fruit?"

    result = selector.select(choices, query)
    assert result.selection_inds == [1, 2, 3]
