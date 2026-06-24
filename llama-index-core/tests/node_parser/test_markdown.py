from llama_index.core.node_parser.file.markdown import MarkdownNodeParser
from llama_index.core.schema import Document


def test_header_splits() -> None:
    markdown_parser = MarkdownNodeParser()

    splits = markdown_parser.get_nodes_from_documents(
        [
            Document(
                text="""# Main Header

Header 1 content

# Header 2
Header 2 content
    """
            )
        ]
    )
    assert len(splits) == 2
    assert splits[0].metadata == {"header_path": "/"}
    assert splits[1].metadata == {"header_path": "/"}
    assert splits[0].text == "# Main Header\n\nHeader 1 content"
    assert splits[1].text == "# Header 2\nHeader 2 content"


def test_header_splits_with_forwardslash() -> None:
    markdown_parser = MarkdownNodeParser(
        header_path_separator="\u203a"
    )  # Unicode for "›", infrequently used char

    splits = markdown_parser.get_nodes_from_documents(
        [
            Document(
                text="""# Main Header

Header 1 content

## FAQ
FAQ content

### 24/7 Support
Support content

#### Contact info
Contact info content
    """
            )
        ]
    )
    assert len(splits) == 4
    assert splits[0].metadata == {"header_path": "›"}
    assert splits[1].metadata == {"header_path": "›Main Header›"}
    assert splits[2].metadata == {"header_path": "›Main Header›FAQ›"}
    assert splits[3].metadata == {"header_path": "›Main Header›FAQ›24/7 Support›"}

    assert splits[0].text == "# Main Header\n\nHeader 1 content"
    assert splits[1].text == "## FAQ\nFAQ content"
    assert splits[2].text == "### 24/7 Support\nSupport content"
    assert splits[3].text == "#### Contact info\nContact info content"


def test_header_splits_with_indented_code_blocks() -> None:
    markdown_parser = MarkdownNodeParser()

    splits = markdown_parser.get_nodes_from_documents(
        [
            Document(
                text="""Some text
# Header 1
## Header 2
### Header 3
```txt
Non indented block code
```
A list begins here:

* Element 1

    ```txt
    # has some indented code, but it's not handled as that.
    ```
* Element 2

```txt
    # also has some code, but unbalanced fences (different number of spaces). Everything after this is considered code block!
 ```

* Element 3
* Element 4
### Another Header 3
 ```txt
# has some wrongly indented fence, and leads to incorrect header detection.
```

## Another Header 2
    """
            )
        ]
    )

    assert len(splits) == 6

    assert splits[0].metadata == {"header_path": "/"}
    assert splits[0].text == "Some text"

    assert splits[1].metadata == {"header_path": "/"}
    assert splits[1].text == "# Header 1"

    assert splits[2].metadata == {"header_path": "/Header 1/"}
    assert splits[2].text == "## Header 2"

    assert splits[3].metadata == {"header_path": "/Header 1/Header 2/"}
    assert splits[3].text.endswith("* Element 4")

    assert splits[4].metadata == {"header_path": "/Header 1/Header 2/"}
    assert splits[4].text.endswith("```")

    assert splits[5].metadata == {"header_path": "/Header 1/"}
    assert splits[5].text == "## Another Header 2"


def test_non_header_splits() -> None:
    markdown_parser = MarkdownNodeParser()

    splits = markdown_parser.get_nodes_from_documents(
        [
            Document(
                text="""# Header 1

#Not a header
Also # not a header
    # Still not a header
    """
            )
        ]
    )
    assert len(splits) == 1


def test_pre_header_content() -> None:
    markdown_parser = MarkdownNodeParser()

    splits = markdown_parser.get_nodes_from_documents(
        [
            Document(
                text="""
pre-header content

# Header 1
Content
## Sub-header
    """
            )
        ]
    )
    assert len(splits) == 3


def test_header_metadata() -> None:
    markdown_parser = MarkdownNodeParser()

    splits = markdown_parser.get_nodes_from_documents(
        [
            Document(
                text="""# Main Header
Content
## Sub-header
Content
### Sub-sub header
Content
# New title
    """
            )
        ]
    )
    assert len(splits) == 4
    assert splits[0].metadata == {"header_path": "/"}
    assert splits[1].metadata == {"header_path": "/Main Header/"}
    assert splits[2].metadata == {"header_path": "/Main Header/Sub-header/"}
    assert splits[3].metadata == {"header_path": "/"}


def test_header_metadata_with_level_jump() -> None:
    markdown_parser = MarkdownNodeParser()

    splits = markdown_parser.get_nodes_from_documents(
        [
            Document(
                text="""# Main Header
Content
### Sub-header
Content
### Sub-sub header
Content
"""
            )
        ]
    )
    assert len(splits) == 3
    assert splits[0].metadata == {"header_path": "/"}
    assert splits[1].metadata == {"header_path": "/Main Header/"}
    assert splits[2].metadata == {"header_path": "/Main Header/"}


def test_tilde_fenced_code_block() -> None:
    """`#` lines inside a ~~~ fenced code block must not be parsed as headers."""
    markdown_parser = MarkdownNodeParser()

    splits = markdown_parser.get_nodes_from_documents(
        [
            Document(
                text="""# Header 1

~~~
# not a header
more code
~~~

Body text
"""
            )
        ]
    )

    # The whole section stays together; the fenced "# not a header" line is not
    # treated as a header, so there is no spurious extra split.
    assert len(splits) == 1
    assert splits[0].metadata == {"header_path": "/"}
    assert "# not a header" in splits[0].text
    assert "Body text" in splits[0].text


def test_mixed_fence_characters() -> None:
    """A ``` line inside a ~~~ block is code content and must not close it."""
    markdown_parser = MarkdownNodeParser()

    splits = markdown_parser.get_nodes_from_documents(
        [
            Document(
                text="""# Header 1

~~~
```
# still inside the tilde fence
~~~

# Header 2
Body
"""
            )
        ]
    )

    # Two sections: the tilde-fenced block belongs to "Header 1" (the inner
    # ``` and "# still inside ..." do not split it), and "Header 2" follows.
    assert len(splits) == 2
    assert splits[0].metadata == {"header_path": "/"}
    assert "# still inside the tilde fence" in splits[0].text
    assert splits[1].text == "# Header 2\nBody"
