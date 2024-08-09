from llama_index.core.node_parser.file.markdown_with_options import (
    MarkdownNodeParserWithOptions,
)
from llama_index.core.schema import Document


def test_markdown_splitter_with_options():
    documents = [
        Document(
            text="""pre-header content
                # How does it work?
                This is a short section.
                ## How do I configure it?
                This is a section containing a large text. Paragraphs are always
                included fully into a node. But, if there are multiple paragraphs
                and the max_node_length is exceeded, the header contents will be split
                across multiple nodes.

                The text continues...
                ## Where can I find it?
                Final node
                """
        )
    ]
    markdown_parser = MarkdownNodeParserWithOptions(max_node_length=50)
    splits = markdown_parser.get_nodes_from_documents(documents)
    assert len(splits) == 7  # How did I configure it? it is split into multiple nodes
    assert len(splits[0].metadata.keys()) == 0
    assert splits[1].metadata["Header 1"] == "How does it work?"
    assert splits[-1].metadata["Header 1"] == "How does it work?"
    assert splits[-1].metadata["Header 2"] == "Where can I find it?"

    # No maximum length
    markdown_parser = MarkdownNodeParserWithOptions()
    splits = markdown_parser.get_nodes_from_documents(documents)
    assert len(splits) == 4  # How did I configure it? is single node
