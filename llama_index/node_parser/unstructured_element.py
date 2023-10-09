"""Unstructured element node parser."""

from typing import List, Optional, Sequence, Callable, Any, Tuple, Dict

from llama_index.bridge.pydantic import Field
from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.interface import NodeParser
from llama_index.node_parser.node_utils import build_nodes_from_splits
from llama_index.schema import BaseNode, TextNode, Document, IndexNode
from llama_index.utils import get_tqdm_iterable
from pydantic import BaseModel
import pandas as pd
from lxml import html
from tqdm import tqdm

from unstructured.partition.html import partition_html


class Element(BaseModel):
    """Element object."""

    id: str
    type: str
    element: Any
    summary: Optional[str] = None
    table: Optional[pd.DataFrame] = None

    class Config:
        arbitrary_types_allowed = True


class TableOutput(BaseModel):
    """Output from analyzing a table."""

    summary: str
    should_keep: bool


def html_to_df(html_str: str) -> pd.DataFrame:
    """Convert HTML to dataframe."""
    tree = html.fromstring(html_str)
    # print(tree.xpath('//table'))
    table_element = tree.xpath("//table")[0]
    rows = table_element.xpath(".//tr")

    data = []
    for row in rows:
        cols = row.xpath(".//td")
        cols = [c.text.strip() if c.text is not None else "" for c in cols]
        data.append(cols)

    df = pd.DataFrame(data[1:], columns=data[0])
    return df


def filter_table(table_element: Any) -> bool:
    """Filter table."""
    table_df = html_to_df(table_element.metadata.text_as_html)
    if len(table_df) <= 1 or len(table_df.columns) <= 1:
        return False
    else:
        return True


def extract_elements(text: str, table_filters: Optional[List[Callable]] = None):
    """Extract elements."""
    table_filters = table_filters or []
    elements = partition_html(text=text)
    output_els = []
    for idx, element in enumerate(elements):
        if "unstructured.documents.html.HTMLTable" in str(type(element)):
            should_keep = all([tf(element) for tf in table_filters])
            if should_keep:
                table_df = html_to_df(str(element.metadata.text_as_html))
                output_els.append(
                    Element(
                        id=f"id_{idx}", type="table", element=element, table=table_df
                    )
                )
            else:
                pass
        else:
            output_els.append(Element(id=f"id_{idx}", type="text", element=element))
    return output_els


def extract_table_summaries(elements: List[Element]) -> None:
    """Go through elements, extract out summaries that are tables."""

    from llama_index.indices.list.base import SummaryIndex

    for element in tqdm(elements):
        if element.type != "table":
            continue
        index = SummaryIndex.from_documents([Document(text=str(element.element))])
        query_engine = index.as_query_engine(output_cls=TableOutput)
        query_str = """\
What is this table about? Give a very concise summary (imagine you are adding a caption), \
and also output whether or not the table should be kept.
"""
        response = query_engine.query(query_str)
        element.summary = response.response.summary


def get_table_elements(elements) -> List[Element]:
    """Get table elements."""
    return [e for e in elements if e.type == "table"]


def get_text_elements(elements) -> List[Element]:
    """Get text elements."""
    return [e for e in elements if e.type == "text"]


def _get_nodes_from_buffer(buffer, node_parser):
    """Get nodes from buffer."""
    doc = Document(text="\n\n".join([t for t in buffer]))
    nodes = node_parser.get_nodes_from_documents([doc])
    return nodes


def get_nodes_and_mappings(elements: List[Element]) -> Tuple[List[BaseNode], Dict]:
    """Get nodes and mappings."""
    node_parser = SimpleNodeParser.from_defaults()

    nodes = []
    node_mappings = {}
    cur_text_el_buffer = []
    for element in elements:
        if element.type == "table":
            # flush text buffer
            if len(cur_text_el_buffer) > 0:
                cur_text_nodes = _get_nodes_from_buffer(cur_text_el_buffer, node_parser)
                nodes.extend(cur_text_nodes)
                cur_text_el_buffer = []

            index_node = IndexNode(
                text=str(element.summary), index_id=(element.id + "_table")
            )
            table_df = element.table
            table_str = table_df.to_string()
            node_mappings[(element.id + "_table")] = TextNode(
                text=table_str, id_=(element.id + "_table")
            )
            nodes.append(index_node)
        else:
            cur_text_el_buffer.append(str(element.element))

    # flush text buffer
    if len(cur_text_el_buffer) > 0:
        cur_text_nodes = _get_nodes_from_buffer(cur_text_el_buffer, node_parser)
        nodes.extend(cur_text_nodes)
        cur_text_el_buffer = []

    return nodes, node_mappings


class UnstructuredElementNodeParser(NodeParser):
    """Unstructured element node parser.

    Splits a document into Text Nodes and Index Nodes corresponding to embedded objects
    (e.g. tables).

    """

    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )

    @classmethod
    def from_defaults(
        cls,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "UnstructuredElementNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        return cls(
            callback_manager=callback_manager,
        )

    def get_nodes_from_node(self, node: TextNode) -> List[BaseNode]:
        """Get nodes from node."""
        elements = extract_elements(node.get_content(), table_filters=[filter_table])
        table_elements = get_table_elements(elements)
        # extract summaries over table elements
        extract_table_summaries(table_elements)
        # convert into nodes
        nodes, node_mappings, other_mappings = get_nodes_and_mappings(elements)
        # will return a list of Nodes and Index Nodes
        return nodes

    def get_nodes_from_documents(
        self,
        documents: Sequence[TextNode],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Parse document into nodes.

        Args:
            documents (Sequence[TextNode]): TextNodes or Documents to parse

        """
        with self.callback_manager.event(
            CBEventType.NODE_PARSING, payload={EventPayload.DOCUMENTS: documents}
        ) as event:
            all_nodes: List[BaseNode] = []
            documents_with_progress = get_tqdm_iterable(
                documents, show_progress, "Parsing documents into nodes"
            )

            for document in documents_with_progress:
                nodes = self.get_nodes_from_node(document)
                all_nodes.extend(nodes)

            event.on_end(payload={EventPayload.NODES: all_nodes})

        return all_nodes
