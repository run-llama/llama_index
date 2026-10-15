"""Default-value tests for SentenceTransformerRerank.

These tests never construct the reranker. They only read the constructor
signature and the Pydantic field, so nothing is downloaded and
`sentence-transformers` is not required to run them.
"""

import inspect

from llama_index.core.postprocessor.sbert_rerank import SentenceTransformerRerank


def test_trust_remote_code_constructor_default_is_false() -> None:
    """The constructor must not enable remote code execution by default.

    `trust_remote_code` is forwarded straight to `CrossEncoder`, so a `True`
    default lets HuggingFace import and execute whatever `modeling_*.py` the
    model repository ships, for any caller that does not pass the argument.
    """
    signature = inspect.signature(SentenceTransformerRerank.__init__)
    assert signature.parameters["trust_remote_code"].default is False


def test_trust_remote_code_constructor_default_matches_field() -> None:
    """The constructor default must agree with the documented field default."""
    signature = inspect.signature(SentenceTransformerRerank.__init__)
    field = SentenceTransformerRerank.model_fields["trust_remote_code"]
    assert signature.parameters["trust_remote_code"].default is field.default
