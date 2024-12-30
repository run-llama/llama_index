from unittest.mock import patch

from llama_index.core.llms import CompletionResponse
from llama_index.core.selectors.llm_selectors import LLMMultiSelector, LLMSingleSelector
from llama_index.core import Settings

from tests.mock_utils.mock_predict import _mock_single_select


def test_llm_single_selector(mock_llm, monkeypatch) -> None:
    selector = LLMSingleSelector.from_defaults()

    monkeypatch.setattr(Settings, "llm", mock_llm)

    with patch.object(
        type(mock_llm),
        "complete",
        return_value=CompletionResponse(text=_mock_single_select()),
    ) as mock_complete:
        result = selector.select(
            choices=["apple", "pear", "peach"], query="what is the best fruit?"
        )
    assert result.ind == 0
    mock_complete.assert_called_once()
    assert mock_complete.call_args.args[0].count("Here is an example") <= 1


def test_llm_multi_selector(patch_llm_predictor) -> None:
    selector = LLMMultiSelector.from_defaults()

    choices = [
        "apple",
        "pear",
        "peach",
    ]
    query = "what is the best fruit?"

    result = selector.select(choices, query)
    assert result.inds == [0, 1, 2]


def test_llm_multi_selector_max_choices(patch_llm_predictor) -> None:
    selector = LLMMultiSelector.from_defaults(max_outputs=2)

    choices = [
        "apple",
        "pear",
        "peach",
    ]
    query = "what is the best fruit?"

    result = selector.select(choices, query)
    assert result.inds == [0, 1]
