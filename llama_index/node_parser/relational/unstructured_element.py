"""Unstructured element node parser."""

from typing import Any, Callable, List, Optional

import pandas as pd

from llama_index.callbacks.base import CallbackManager
from llama_index.node_parser.relational.base_element import (
    DEFAULT_SUMMARY_QUERY_STR,
    BaseElementNodeParser,
    Element,
)
from llama_index.schema import BaseNode, TextNode


def html_to_df(html_str: str) -> pd.DataFrame:
    """Convert HTML to dataframe."""
    from lxml import html

    tree = html.fromstring(html_str)
    table_element = tree.xpath("//table")[0]
    rows = table_element.xpath(".//tr")

    data = []
    for row in rows:
        cols = row.xpath(".//td")
        cols = [c.text.strip() if c.text is not None else "" for c in cols]
        data.append(cols)

    # Check if the table is empty
    if len(data) == 0:
        return None

    # Check if the all rows have the same number of columns
    if not all(len(row) == len(data[0]) for row in data):
        return None

    return pd.DataFrame(data[1:], columns=data[0])


class UnstructuredElementNodeParser(BaseElementNodeParser):
    """Unstructured element node parser.

    Splits a document into Text Nodes and Index Nodes corresponding to embedded objects
    (e.g. tables).

    """

    def __init__(
        self,
        callback_manager: Optional[CallbackManager] = None,
        llm: Optional[Any] = None,
        summary_query_str: str = DEFAULT_SUMMARY_QUERY_STR,
    ) -> None:
        """Initialize."""
        try:
            import lxml  # noqa
            import unstructured  # noqa
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
        return self.get_nodes_from_elements(elements)

    def extract_elements(
        self, text: str, table_filters: Optional[List[Callable]] = None, **kwargs: Any
    ) -> List[Element]:
        """Extract elements from text."""
        from unstructured.partition.html import partition_html

        table_filters = table_filters or []
        elements = partition_html(text=text)
        output_els = []
        for idx, element in enumerate(elements):
            if "unstructured.documents.html.HTMLTable" in str(type(element)):
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
                    # if not a table, keep it as Text as we don't want to loose context
                    from unstructured.documents.html import HTMLText

                    newElement = HTMLText(str(element), tag=element.tag)
                    output_els.append(
                        Element(id=f"id_{idx}", type="text", element=newElement)
                    )
            else:
                output_els.append(Element(id=f"id_{idx}", type="text", element=element))
        return output_els

    def filter_table(self, table_element: Any) -> bool:
        """Filter tables."""
        table_df = html_to_df(table_element.metadata.text_as_html)

        # check if table_df is not None, has more than one row, and more than one column
        return table_df is not None and not table_df.empty and len(table_df.columns) > 1
