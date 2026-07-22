import llama_index.core.node_parser.file.markdown as markdown_module
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


# Regression test for quadratic parsing: MarkdownNodeParser must call
# build_nodes_from_splits once for the whole document instead of once per
# header section. Calling it per section recomputes the source document hash
# (ref_doc.as_related_node_info()) for every section, making parsing
# O(sections * document_size) -- quadratic on header-dense documents.
def test_builds_all_nodes_in_a_single_call(monkeypatch) -> None:
    call_sizes = []
    original = markdown_module.build_nodes_from_splits

    def spy(text_splits, document, *args, **kwargs):
        call_sizes.append(len(text_splits))
        return original(text_splits, document, *args, **kwargs)

    monkeypatch.setattr(markdown_module, "build_nodes_from_splits", spy)

    parser = MarkdownNodeParser()
    splits = parser.get_nodes_from_documents(
        [Document(text="# A\nalpha\n\n# B\nbeta\n\n# C\ngamma\n")]
    )

    assert len(splits) == 3
    # Exactly one batched call covering all sections, not one call per section.
    assert call_sizes == [3]
    assert splits[0].text == "# A\nalpha"
    assert splits[1].text == "# B\nbeta"
    assert splits[2].text == "# C\ngamma"
    assert splits[0].metadata == {"header_path": "/"}
    assert splits[1].metadata == {"header_path": "/"}
    assert splits[2].metadata == {"header_path": "/"}
