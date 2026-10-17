import pytest

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


def test_single_selector_raises_on_zero_indexed_answer() -> None:
    from llama_index.core.selectors.llm_selectors import (
        _structured_output_to_selector_result,
    )
    from llama_index.core.output_parsers.base import StructuredOutput
    from llama_index.core.output_parsers.selection import Answer

    output = StructuredOutput(
        raw_output="", parsed_output=[Answer(choice=0, reason="first")]
    )
    with pytest.raises(ValueError, match="out-of-range"):
        _structured_output_to_selector_result(output, 3)


def test_single_selector_raises_on_over_range_answer() -> None:
    from llama_index.core.selectors.llm_selectors import (
        _structured_output_to_selector_result,
    )
    from llama_index.core.output_parsers.base import StructuredOutput
    from llama_index.core.output_parsers.selection import Answer

    output = StructuredOutput(
        raw_output="", parsed_output=[Answer(choice=5, reason="last")]
    )
    with pytest.raises(ValueError, match="out-of-range"):
        _structured_output_to_selector_result(output, 3)


def test_multi_selector_drops_out_of_range_answers() -> None:
    from llama_index.core.selectors.llm_selectors import (
        _structured_output_to_selector_result,
    )
    from llama_index.core.output_parsers.base import StructuredOutput
    from llama_index.core.output_parsers.selection import Answer

    output = StructuredOutput(
        raw_output="",
        parsed_output=[
            Answer(choice=1, reason="a"),
            Answer(choice=0, reason="zero-indexed"),
            Answer(choice=4, reason="over-range"),
        ],
    )
    result = _structured_output_to_selector_result(output, 3)
    assert result.inds == [0]
    assert result.selections[0].reason == "a"


def test_pydantic_selector_drops_out_of_range_selections() -> None:
    from llama_index.core.selectors.pydantic_selectors import (
        _pydantic_output_to_selector_result,
    )
    from llama_index.core.base.base_selector import MultiSelection, SingleSelection

    result = _pydantic_output_to_selector_result(
        MultiSelection(
            selections=[
                SingleSelection(index=2, reason="ok"),
                SingleSelection(index=99, reason="over-range"),
            ]
        ),
        3,
    )
    assert result.inds == [1]
    assert result.selections[0].reason == "ok"


def test_pydantic_single_selector_raises_on_zero_indexed_answer() -> None:
    from llama_index.core.selectors.pydantic_selectors import (
        _pydantic_output_to_selector_result,
    )
    from llama_index.core.base.base_selector import SingleSelection

    with pytest.raises(ValueError, match="out-of-range"):
        _pydantic_output_to_selector_result(
            SingleSelection(index=0, reason="zero-indexed"), 3
        )
