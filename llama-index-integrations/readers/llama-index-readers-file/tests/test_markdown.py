from llama_index.readers.file import MarkdownReader


def test_parse_markdown_starting_with_header() -> None:
    reader = MarkdownReader()
    markdown_text = "# ABC\nabc\n# DEF\ndef"
    expected_tups = [("ABC", "abc"), ("DEF", "def")]
    assert reader.markdown_to_tups(markdown_text) == expected_tups


def test_parse_markdown_with_text_before_first_header() -> None:
    reader = MarkdownReader()
    markdown_text = "abc\n# ABC\ndef"
    expected_tups = [(None, "abc"), ("ABC", "def")]
    assert reader.markdown_to_tups(markdown_text) == expected_tups


def test_parse_markdown_with_empty_lines_before_first_header() -> None:
    reader = MarkdownReader()
    markdown_text = "\n\n\n# ABC\ndef"
    expected_tups = [(None, "\n\n"), ("ABC", "def")]
    assert reader.markdown_to_tups(markdown_text) == expected_tups


def test_parse_markdown_with_no_headers() -> None:
    reader = MarkdownReader()
    markdown_text = "abc\ndef"
    expected_tups = [(None, "abc\ndef")]
    assert reader.markdown_to_tups(markdown_text) == expected_tups


def test_parse_markdown_with_only_headers() -> None:
    reader = MarkdownReader()
    markdown_text = "# ABC\n# DEF"
    expected_tups = [("ABC", ""), ("DEF", "")]
    assert reader.markdown_to_tups(markdown_text) == expected_tups


def test_parse_empty_markdown() -> None:
    reader = MarkdownReader()
    markdown_text = ""
    expected_tups = [(None, "")]
    assert reader.markdown_to_tups(markdown_text) == expected_tups


def test_parse_omits_trailing_newline_before_new_header() -> None:
    reader = MarkdownReader()

    markdown_text = ("\n" * 4) + "# ABC\nabc"
    expected_tups = [(None, "\n" * 3), ("ABC", "abc")]
    assert reader.markdown_to_tups(markdown_text) == expected_tups

    markdown_text = ("\n" * 4) + "# ABC\nabc\n"
    expected_tups = [(None, "\n" * 3), ("ABC", "abc\n")]
    assert reader.markdown_to_tups(markdown_text) == expected_tups

    markdown_text = "\n" * 4
    expected_tups = [(None, "\n" * 4)]
    assert reader.markdown_to_tups(markdown_text) == expected_tups
