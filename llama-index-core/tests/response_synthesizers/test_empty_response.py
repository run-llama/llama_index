from llama_index.core.response_synthesizers import SimpleSummarize
from llama_index.core.response_synthesizers.factory import get_response_synthesizer


def test_explicit_empty_string_response_is_respected() -> None:
    synthesizer = SimpleSummarize(empty_response="")

    assert synthesizer._empty_response == ""


def test_get_response_synthesizer_forwards_empty_response() -> None:
    synthesizer = get_response_synthesizer(response_mode="simple", empty_response="")

    assert synthesizer._empty_response == ""
