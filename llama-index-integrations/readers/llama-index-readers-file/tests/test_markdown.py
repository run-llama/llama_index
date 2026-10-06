from pathlib import Path
from typing import List, Tuple

import pytest

from llama_index.readers.file.markdown.base import MarkdownReader


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
    expected_tups = [("ABC", "def")]
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


def test_parse_markdown_with_headers_in_code_block() -> None:
    reader = MarkdownReader()
    markdown_text = """# ABC
```python
# This is a comment
print("hello")
```
# DEF
"""
    expected_tups = [
        ("ABC", '```python\n# This is a comment\nprint("hello")\n```'),
        ("DEF", ""),
    ]
    assert reader.markdown_to_tups(markdown_text) == expected_tups


def test_parse_empty_markdown() -> None:
    reader = MarkdownReader()
    markdown_text = ""
    expected_tups = []
    assert reader.markdown_to_tups(markdown_text) == expected_tups


def test_parse_omits_trailing_newline_before_new_header() -> None:
    reader = MarkdownReader()

    markdown_text = ("\n" * 4) + "# ABC\nabc"
    expected_tups = [("ABC", "abc")]
    assert reader.markdown_to_tups(markdown_text) == expected_tups

    markdown_text = ("\n" * 4) + "# ABC\nabc\n"
    expected_tups = [("ABC", "abc")]
    assert reader.markdown_to_tups(markdown_text) == expected_tups

    markdown_text = "\n" * 4
    expected_tups = []
    assert reader.markdown_to_tups(markdown_text) == expected_tups


def test_multiple_class_titles_parse() -> None:
    reader = MarkdownReader()
    markdown_text = """
# Main Title (Level 1)

## Section 1: Introduction (Level 2)

### Subsection 1.1: Background (Level 3)

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia arcu eget nulla fermentum, et suscipit justo volutpat.

### Subsection 1.2: Objective (Level 3)

Curabitur non nulla sit amet nisl tempus convallis quis ac lectus. Integer posuere erat a ante venenatis dapibus posuere velit aliquet.

## Section 2: Methodology (Level 2)

### Subsection 2.1: Approach (Level 3)

Mauris blandit aliquet elit, eget tincidunt nibh pulvinar a. Pellentesque in ipsum id orci porta dapibus.

### Subsection 2.2: Tools and Techniques (Level 3)

Donec rutrum congue leo eget malesuada. Vivamus suscipit tortor eget felis porttitor volutpat.

#### Sub-subsection 2.2.1: Tool 1 (Level 4)

Donec sollicitudin molestie malesuada.

#### Sub-subsection 2.2.2: Tool 2 (Level 4)

Proin eget tortor risus. Cras ultricies ligula sed magna dictum porta.

## Section 3: Results (Level 2)

### Subsection 3.1: Data Analysis (Level 3)

Sed porttitor lectus nibh. Donec rutrum congue leo eget malesuada.

### Subsection 3.2: Findings (Level 3)

Curabitur arcu erat, accumsan id imperdiet et, porttitor at sem.
    """
    expected_tups = [
        (
            "Main Title (Level 1) Section 1: Introduction (Level 2) Subsection 1.1: Background (Level 3)",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia arcu eget nulla fermentum, et suscipit justo volutpat.",
        ),
        (
            "Main Title (Level 1) Section 1: Introduction (Level 2) Subsection 1.2: Objective (Level 3)",
            "Curabitur non nulla sit amet nisl tempus convallis quis ac lectus. Integer posuere erat a ante venenatis dapibus posuere velit aliquet.",
        ),
        (
            "Main Title (Level 1) Section 2: Methodology (Level 2) Subsection 2.1: Approach (Level 3)",
            "Mauris blandit aliquet elit, eget tincidunt nibh pulvinar a. Pellentesque in ipsum id orci porta dapibus.",
        ),
        (
            "Main Title (Level 1) Section 2: Methodology (Level 2) Subsection 2.2: Tools and Techniques (Level 3) Sub-subsection 2.2.1: Tool 1 (Level 4)",
            "Donec rutrum congue leo eget malesuada. Vivamus suscipit tortor eget felis porttitor volutpat.\nDonec sollicitudin molestie malesuada.",
        ),
        (
            "Main Title (Level 1) Section 2: Methodology (Level 2) Subsection 2.2: Tools and Techniques (Level 3) Sub-subsection 2.2.2: Tool 2 (Level 4)",
            "Proin eget tortor risus. Cras ultricies ligula sed magna dictum porta.",
        ),
        (
            "Main Title (Level 1) Section 3: Results (Level 2) Subsection 3.1: Data Analysis (Level 3)",
            "Sed porttitor lectus nibh. Donec rutrum congue leo eget malesuada.",
        ),
        (
            "Main Title (Level 1) Section 3: Results (Level 2) Subsection 3.2: Findings (Level 3)",
            "Curabitur arcu erat, accumsan id imperdiet et, porttitor at sem.",
        ),
    ]

    assert reader.markdown_to_tups(markdown_text) == expected_tups


def test_blank_lines_in_markdown() -> None:
    reader = MarkdownReader()
    markdown_text = """


    """
    expected_tups = []
    assert reader.markdown_to_tups(markdown_text) == expected_tups


@pytest.mark.parametrize(
    ("markdown_text", "expected_tups"),
    [
        (
            "# Root\n### Deep\nalpha\n## Sibling\nbeta",
            [("Root Deep", "alpha"), ("Root Sibling", "beta")],
        ),
        (
            "### First\nalpha\n## Second\nbeta",
            [("First", "alpha"), ("Second", "beta")],
        ),
        (
            "### First\nalpha\n# Second\nbeta",
            [("First", "alpha"), ("Second", "beta")],
        ),
        (
            "# Root\n#### Deep\nalpha\n### Middle\nbeta\n## Shallow\ngamma",
            [
                ("Root Deep", "alpha"),
                ("Root Middle", "beta"),
                ("Root Shallow", "gamma"),
            ],
        ),
        (
            "# Root\n### Deep\n## Sibling\nbeta",
            [("Root Deep", ""), ("Root Sibling", "beta")],
        ),
    ],
)
def test_parse_markdown_with_skipped_heading_levels(
    markdown_text: str, expected_tups: List[Tuple[str, str]]
) -> None:
    assert MarkdownReader().markdown_to_tups(markdown_text) == expected_tups


def test_load_markdown_with_skipped_heading_levels(tmp_path: Path) -> None:
    path = tmp_path / "handbook.md"
    path.write_text(
        "# Handbook\n### Authentication\nRotate credentials monthly.\n"
        "## Billing\nRefunds are available within thirty days.",
        encoding="utf-8",
    )
    documents = MarkdownReader(separator=" / ").load_data(
        str(path), extra_info={"source": "handbook"}
    )
    assert [document.text for document in documents] == [
        "\n\nHandbook / Authentication\nRotate credentials monthly.",
        "\n\nHandbook / Billing\nRefunds are available within thirty days.",
    ]
    assert all(document.metadata == {"source": "handbook"} for document in documents)
