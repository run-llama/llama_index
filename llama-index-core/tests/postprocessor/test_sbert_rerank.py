"""Tests for SentenceTransformerRerank."""

import builtins
import sys
from typing import Any

import pytest


def test_missing_dependency_message_is_a_single_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The ImportError used to be raised with two arguments, so str(e) was a tuple repr.

    Anything logging or displaying it showed quotes, a comma and parentheses around the
    sentence, including the pip command the message exists to hand the user.
    """
    from llama_index.core.postprocessor.sbert_rerank import SentenceTransformerRerank

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sentence_transformers":
            raise ImportError("mocked missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError) as exc_info:
        SentenceTransformerRerank()

    message = str(exc_info.value)
    assert message == (
        "Cannot import sentence-transformers or torch package, "
        "please `pip install torch sentence-transformers`"
    )
    assert len(exc_info.value.args) == 1
