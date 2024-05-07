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
    assert splits[0].metadata == {"Header_1": "Main Header"}
    assert splits[1].metadata == {"Header_1": "Header 2"}
    assert splits[0].text == "Main Header\n\nHeader 1 content"
    assert splits[1].text == "Header 2\nHeader 2 content"


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

    assert splits[0].metadata == {}
    assert splits[0].text == "Some text"

    assert splits[1].metadata == {"Header_1": "Header 1"}
    assert splits[1].text == "Header 1"

    assert splits[2].metadata == {"Header_1": "Header 1", "Header_2": "Header 2"}
    assert splits[2].text == "Header 2"

    assert splits[3].metadata == {
        "Header_1": "Header 1",
        "Header_2": "Header 2",
        "Header_3": "Header 3",
    }
    assert splits[3].text.endswith("* Element 4")

    assert splits[4].metadata == {
        "Header_1": "Header 1",
        "Header_2": "Header 2",
        "Header_3": "Another Header 3",
    }
    assert splits[4].text.endswith("```")

    assert splits[5].metadata == {
        "Header_1": "Header 1",
        "Header_2": "Another Header 2",
    }
    assert splits[5].text == "Another Header 2"


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
    assert splits[0].metadata == {"Header_1": "Main Header"}
    assert splits[1].metadata == {"Header_1": "Main Header", "Header_2": "Sub-header"}
    assert splits[2].metadata == {
        "Header_1": "Main Header",
        "Header_2": "Sub-header",
        "Header_3": "Sub-sub header",
    }
    assert splits[3].metadata == {"Header_1": "New title"}
