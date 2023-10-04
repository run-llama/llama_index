from llama_index.node_parser.file.markdown import MarkdownNodeParser
from llama_index.schema import Document


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
    assert splits[0].metadata == {"Header 1": "Main Header"}
    assert splits[1].metadata == {"Header 1": "Header 2"}
    assert splits[0].text == "Main Header\n\nHeader 1 content"
    assert splits[1].text == "Header 2\nHeader 2 content"


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
    assert splits[0].metadata == {"Header 1": "Main Header"}
    assert splits[1].metadata == {"Header 1": "Main Header", "Header 2": "Sub-header"}
    assert splits[2].metadata == {
        "Header 1": "Main Header",
        "Header 2": "Sub-header",
        "Header 3": "Sub-sub header",
    }
    assert splits[3].metadata == {"Header 1": "New title"}
