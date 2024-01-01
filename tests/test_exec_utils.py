from llama_index.exec_utils import _contains_dunder_calls


def test_contains_dunder_calls():
    assert not _contains_dunder_calls(
        "def _a(b): pass"
    ), "definition of dunder function"
    assert _contains_dunder_calls("a = _b(c)"), "call to dunder function"
    assert not _contains_dunder_calls("a = b(c)"), "call to non-dunder function"
