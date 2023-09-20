from llama_index.text_splitter.markdown_splitter import MarkdownSplitter


def test_header_splits() -> None:
    markdown_splitter = MarkdownSplitter()

    splits = markdown_splitter.split_text(
        """# Header 1

Header 1 content

# Header 2
Header 2 content
    """
    )
    print(splits)
    assert len(splits) == 2


def test_non_header_splits() -> None:
    markdown_splitter = MarkdownSplitter()

    splits = markdown_splitter.split_text(
        """# Header 1

#Not a header
Also # not a header
    # Still not a header
    """
    )
    assert len(splits) == 1


def test_pre_header_content() -> None:
    markdown_splitter = MarkdownSplitter()

    splits = markdown_splitter.split_text(
        """
pre-header content

# Header 1
Content
## Sub-header
    """
    )
    assert len(splits) == 3
