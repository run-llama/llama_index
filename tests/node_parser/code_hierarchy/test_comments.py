from llama_index.node_parser.code_hierarchy import _generate_comment_line


def test_generate_comment_line_python() -> None:
    assert (
        _generate_comment_line("Python", "This is a Python comment")
        == "# This is a Python comment"
    )


def test_generate_comment_line_c_lowercase() -> None:
    assert (
        _generate_comment_line("c", "This is a C comment") == "// This is a C comment"
    )


def test_generate_comment_line_java() -> None:
    assert (
        _generate_comment_line("Java", "This is a Java comment")
        == "// This is a Java comment"
    )


def test_generate_comment_line_html() -> None:
    assert (
        _generate_comment_line("HTML", "This is an HTML comment")
        == "<!-- This is an HTML comment -->"
    )


def test_generate_comment_line_unknown() -> None:
    assert (
        _generate_comment_line("Unknown", "This is an unknown language comment")
        == "🦙This is an unknown language comment🦙"
    )


def test_generate_comment_line_unknown_random() -> None:
    assert (
        _generate_comment_line("asdf", "This is an unknown language comment")
        == "🦙This is an unknown language comment🦙"
    )
