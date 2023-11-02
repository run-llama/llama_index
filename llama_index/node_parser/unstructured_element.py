"""Unstructured element node parser."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import pandas as pd
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from llama_index.bridge.pydantic import Field
from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.llms.openai import LLM, OpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.interface import NodeParser
from llama_index.response.schema import PydanticResponse
from llama_index.schema import BaseNode, Document, IndexNode, TextNode
from llama_index.utils import get_tqdm_iterable


class TableColumnOutput(BaseModel):
    """Output from analyzing a table column."""

    col_name: str
    col_type: str
    summary: Optional[str] = None

    def __str__(self) -> str:
        """Convert to string representation."""
        return (
            f"Column: {self.col_name}\nType: {self.col_type}\nSummary: {self.summary}"
        )


class TableOutput(BaseModel):
    """Output from analyzing a table."""

    summary: str
    columns: List[TableColumnOutput]


class Element(BaseModel):
    """Element object."""

    id: str
    type: str
    element: Any
    table_output: Optional[TableOutput] = None
    table: Optional[pd.DataFrame] = None

    class Config:
        arbitrary_types_allowed = True


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

    return pd.DataFrame(data[1:], columns=data[0])


def filter_table(table_element: Any) -> bool:
    """Filter table."""
    table_df = html_to_df(table_element.metadata.text_as_html)
    return len(table_df) > 1 and len(table_df.columns) > 1


def extract_elements(
    text: str, table_filters: Optional[List[Callable]] = None
) -> List[Element]:
    """Extract elements."""
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
                        id=f"id_{idx}", type="table", element=element, table=table_df
                    )
                )
            else:
                pass
        else:
            output_els.append(Element(id=f"id_{idx}", type="text", element=element))
    return output_els


def extract_table_summaries(
    elements: List[Element], llm: Optional[Any], summary_query_str: str
) -> None:
    """Go through elements, extract out summaries that are tables."""
    from llama_index.indices.list.base import SummaryIndex
    from llama_index.indices.service_context import ServiceContext

    llm = llm or OpenAI()
    llm = cast(LLM, llm)

    service_context = ServiceContext.from_defaults(llm=llm)
    for element in tqdm(elements):
        if element.type != "table":
            continue
        index = SummaryIndex.from_documents(
            [Document(text=str(element.element))], service_context=service_context
        )
        query_engine = index.as_query_engine(output_cls=TableOutput)
        try:
            response = query_engine.query(summary_query_str)
            element.table_output = cast(PydanticResponse, response).response
        except ValidationError:
            # There was a pydantic validation error, so we will run with text completion
            # fill in the summary and leave other fields blank
            query_engine = index.as_query_engine()
            response_txt = str(query_engine.query(summary_query_str))
            element.table_output = TableOutput(summary=response_txt, columns=[])


def get_table_elements(elements: List[Element]) -> List[Element]:
    """Get table elements."""
    return [e for e in elements if e.type == "table"]


def get_text_elements(elements: List[Element]) -> List[Element]:
    """Get text elements."""
    return [e for e in elements if e.type == "text"]


def _get_nodes_from_buffer(
    buffer: List[str], node_parser: NodeParser
) -> List[BaseNode]:
    """Get nodes from buffer."""
    doc = Document(text="\n\n".join(list(buffer)))
    return node_parser.get_nodes_from_documents([doc])


def get_nodes_from_elements(elements: List[Element]) -> List[BaseNode]:
    """Get nodes and mappings."""
    node_parser = SimpleNodeParser.from_defaults()

    nodes = []
    cur_text_el_buffer: List[str] = []
    for element in elements:
        if element.type == "table":
            # flush text buffer
            if len(cur_text_el_buffer) > 0:
                cur_text_nodes = _get_nodes_from_buffer(cur_text_el_buffer, node_parser)
                nodes.extend(cur_text_nodes)
                cur_text_el_buffer = []

            table_output = cast(TableOutput, element.table_output)
            table_df = cast(pd.DataFrame, element.table)

            table_id = element.id + "_table"
            table_ref_id = element.id + "_table_ref"
            # TODO: figure out what to do with columns
            # NOTE: right now they're excluded from embedding
            col_schema = "\n\n".join([str(col) for col in table_output.columns])
            index_node = IndexNode(
                text=str(table_output.summary),
                metadata={"col_schema": col_schema},
                excluded_embed_metadata_keys=["col_schema"],
                id_=table_ref_id,
                index_id=table_id,
            )
            table_str = table_df.to_string()
            text_node = TextNode(
                text=table_str,
                id_=table_id,
            )
            nodes.extend([index_node, text_node])
        else:
            cur_text_el_buffer.append(str(element.element))

    # flush text buffer
    if len(cur_text_el_buffer) > 0:
        cur_text_nodes = _get_nodes_from_buffer(cur_text_el_buffer, node_parser)
        nodes.extend(cur_text_nodes)
        cur_text_el_buffer = []

    return nodes


DEFAULT_SUMMARY_QUERY_STR = """\
What is this table about? Give a very concise summary (imagine you are adding a caption), \
and also output whether or not the table should be kept.\
"""


class UnstructuredElementNodeParser(NodeParser):
    """Unstructured element node parser.

    Splits a document into Text Nodes and Index Nodes corresponding to embedded objects
    (e.g. tables).

    """

    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )
    llm: Optional[LLM] = Field(
        default=None, description="LLM model to use for summarization."
    )
    summary_query_str: str = Field(
        default=DEFAULT_SUMMARY_QUERY_STR,
        description="Query string to use for summarization.",
    )

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
                "You must install the `unstructured` and `lxml` package to use this node parser."
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
        extract_table_summaries(table_elements, self.llm, self.summary_query_str)

        # convert into nodes
        # will return a list of Nodes and Index Nodes
        return get_nodes_from_elements(elements)

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

    def get_base_nodes_and_mappings(
        self, nodes: List[BaseNode]
    ) -> Tuple[List[BaseNode], Dict]:
        """Get base nodes and mappings.

        Given a list of nodes and IndexNode objects, return the base nodes and a mapping
        from index id to child nodes (which are excluded from the base nodes).

        """
        node_dict = {node.node_id: node for node in nodes}

        node_mappings = {}
        base_nodes = []

        # first map index nodes to their child nodes
        nonbase_node_ids = set()
        for node in nodes:
            if isinstance(node, IndexNode):
                node_mappings[node.index_id] = node_dict[node.index_id]
                nonbase_node_ids.add(node.index_id)
            else:
                pass

        # then add all nodes that are not children of index nodes
        for node in nodes:
            if node.node_id not in nonbase_node_ids:
                base_nodes.append(node)

        return base_nodes, node_mappings
