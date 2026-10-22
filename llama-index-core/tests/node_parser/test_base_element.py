import pytest

from llama_index.core.node_parser.relational.base_element import Element, TableOutput
from llama_index.core.node_parser.relational.markdown_element import (
    MarkdownElementNodeParser,
)
from llama_index.core.schema import MetadataMode, TextNode


@pytest.mark.parametrize("exclude_source_metadata", [False, True])
def test_table_nodes_preserve_metadata_exclusions(
    exclude_source_metadata: bool,
) -> None:
    source = TextNode(
        text="<table><tr><td>42</td></tr></table>",
        metadata={"source": "report", "internal": "tracking-id"},
        excluded_embed_metadata_keys=["internal"] if exclude_source_metadata else [],
        excluded_llm_metadata_keys=["internal"] if exclude_source_metadata else [],
    )
    element = Element(
        id="table",
        type="table_text",
        element=source.text,
        table_output=TableOutput(summary="Reported value", columns=[]),
    )

    nodes = MarkdownElementNodeParser().get_nodes_from_elements(
        [element], node_inherited=source
    )
    index_node, table_node = nodes

    assert "col_schema" in index_node.excluded_embed_metadata_keys
    for mode in (MetadataMode.EMBED, MetadataMode.LLM):
        metadata = table_node.get_metadata_str(mode=mode)
        assert "table_df" not in metadata
        assert "table_summary" not in metadata
        assert "source: report" in metadata
        assert ("internal: tracking-id" not in metadata) == exclude_source_metadata

    for node in nodes:
        assert node.metadata["source"] == "report"
        assert (
            "internal" in node.excluded_embed_metadata_keys
        ) == exclude_source_metadata
        assert (
            "internal" in node.excluded_llm_metadata_keys
        ) == exclude_source_metadata

    table_node.excluded_embed_metadata_keys.append("source")
    table_node.excluded_llm_metadata_keys.append("source")
    assert "source" not in source.excluded_embed_metadata_keys
    assert "source" not in source.excluded_llm_metadata_keys
    assert "source" not in index_node.excluded_embed_metadata_keys
    assert "source" not in index_node.excluded_llm_metadata_keys
