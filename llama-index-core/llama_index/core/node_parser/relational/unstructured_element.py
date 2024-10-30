"""Unstructured element node parser."""

from typing import Any, Callable, List, Optional, Dict


from llama_index.core.bridge.pydantic import Field

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.relational.base_element import (
    DEFAULT_SUMMARY_QUERY_STR,
    BaseElementNodeParser,
    Element,
)
from llama_index.core.schema import BaseNode, NodeRelationship, TextNode
from llama_index.core.node_parser.relational.utils import html_to_df


class UnstructuredElementNodeParser(BaseElementNodeParser):
    """Unstructured element node parser.

    Splits a document into Text Nodes and Index Nodes corresponding to embedded objects
    (e.g. tables).

    """

    partitioning_parameters: Optional[Dict[str, Any]] = Field(
        default={},
        description="Extra dictionary representing parameters of the partitioning process.",
    )

    def __init__(
        self,
        callback_manager: Optional[CallbackManager] = None,
        llm: Optional[Any] = None,
        summary_query_str: str = DEFAULT_SUMMARY_QUERY_STR,
        partitioning_parameters: Optional[Dict[str, Any]] = {},
    ) -> None:
        """Initialize."""
        try:
            import lxml  # noqa  # pants: no-infer-dep
            import unstructured  # noqa  # pants: no-infer-dep
        except ImportError:
            raise ImportError(
                "You must install the `unstructured` and `lxml` "
                "package to use this node parser."
            )
        callback_manager = callback_manager or CallbackManager([])

        return super().__init__(
            callback_manager=callback_manager,
            llm=llm,
            summary_query_str=summary_query_str,
            partitioning_parameters=partitioning_parameters,
        )

    @classmethod
    def class_name(cls) -> str:
        return "UnstructuredElementNodeParser"

    def get_nodes_from_node(self, node: TextNode) -> List[BaseNode]:
        """Get nodes from node."""
        elements = self.extract_elements(
            node.get_content(), table_filters=[self.filter_table]
        )
        table_elements = self.get_table_elements(elements)
        # extract summaries over table elements
        self.extract_table_summaries(table_elements)
        # convert into nodes
        # will return a list of Nodes and Index Nodes
        nodes = self.get_nodes_from_elements(
            elements, node, ref_doc_text=node.get_content()
        )

        source_document = node.source_node or node.as_related_node_info()
        for n in nodes:
            n.relationships[NodeRelationship.SOURCE] = source_document
            n.metadata.update(node.metadata)
        return nodes

    async def aget_nodes_from_node(self, node: TextNode) -> List[BaseNode]:
        """Get nodes from node."""
        elements = self.extract_elements(
            node.get_content(), table_filters=[self.filter_table]
        )
        table_elements = self.get_table_elements(elements)
        # extract summaries over table elements
        await self.aextract_table_summaries(table_elements)
        # convert into nodes
        # will return a list of Nodes and Index Nodes
        nodes = self.get_nodes_from_elements(
            elements, node, ref_doc_text=node.get_content()
        )

        source_document = node.source_node or node.as_related_node_info()
        for n in nodes:
            n.relationships[NodeRelationship.SOURCE] = source_document
            n.metadata.update(node.metadata)
        return nodes

    def extract_elements(
        self, text: str, table_filters: Optional[List[Callable]] = None, **kwargs: Any
    ) -> List[Element]:
        """Extract elements from text."""
        from unstructured.partition.html import partition_html  # pants: no-infer-dep

        table_filters = table_filters or []
        elements = partition_html(text=text, **self.partitioning_parameters)
        output_els = []
        for idx, element in enumerate(elements):
            if "unstructured.documents.elements.Table" in str(type(element)):
                should_keep = all(tf(element) for tf in table_filters)
                if should_keep:
                    table_df = html_to_df(str(element.metadata.text_as_html))
                    output_els.append(
                        Element(
                            id=f"id_{idx}",
                            type="table",
                            element=element,
                            table=table_df,
                        )
                    )
                else:
                    # if not a table, keep it as Text as we don't want to lose context
                    from unstructured.documents.elements import Text

                    new_element = Text(str(element))
                    output_els.append(
                        Element(id=f"id_{idx}", type="text", element=new_element)
                    )
            else:
                output_els.append(Element(id=f"id_{idx}", type="text", element=element))
        return output_els

    def filter_table(self, table_element: Any) -> bool:
        """Filter tables."""
        table_df = html_to_df(table_element.metadata.text_as_html)

        # check if table_df is not None, has more than one row, and more than one column
        return table_df is not None and not table_df.empty and len(table_df.columns) > 1
