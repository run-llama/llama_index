from llama_index.core.node_parser.text.utils import split_text_keep_separator


def test_split_text_keep_separator_appends_separator() -> None:
    text = "Hello. World. Bye."

    assert split_text_keep_separator(text, ".") == [
        "Hello.",
        " World.",
        " Bye.",
    ]


def test_split_text_keep_separator_preserves_text_with_consecutive_separators() -> None:
    text = "a..b"
    chunks = split_text_keep_separator(text, ".")

    assert chunks == ["a.", ".", "b"]
    assert "".join(chunks) == text
