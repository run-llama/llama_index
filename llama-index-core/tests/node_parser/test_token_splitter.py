import logging

from llama_index.core.node_parser.text.token import TokenTextSplitter


def test_oversized_split_warning_is_complete(caplog):
    """A split larger than chunk_size logs a single, fully-formed warning.

    Previously the warning was built from two f-string arguments, so logging
    treated the second as a (dropped) %-format arg and getMessage() failed.
    """
    splitter = TokenTextSplitter(chunk_size=5, chunk_overlap=0)
    oversized = "word " * 30  # far more than 5 tokens

    with caplog.at_level(logging.WARNING):
        splitter._merge([oversized], chunk_size=5)

    warnings = [
        r.getMessage()  # would raise TypeError if the args bug were present
        for r in caplog.records
        if r.levelno == logging.WARNING
    ]
    assert any("larger than chunk size" in m for m in warnings), warnings
