from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import pandas as pd
from tqdm import tqdm

from llama_index.bridge.pydantic import BaseModel, Field, ValidationError
from llama_index.callbacks.base import CallbackManager
from llama_index.core.response.schema import PydanticResponse
from llama_index.llms.llm import LLM
from llama_index.llms.openai import OpenAI
from llama_index.node_parser.interface import NodeParser
from llama_index.schema import BaseNode, Document, IndexNode, TextNode
from llama_index.utils import get_tqdm_iterable

DEFAULT_SUMMARY_QUERY_STR = """\
What is this table about? Give a very concise summary (imagine you are adding a caption), \
and also output whether or not the table should be kept.\
"""


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
    title_level: Optional[int] = None
    table_output: Optional[TableOutput] = None
    table: Optional[pd.DataFrame] = None

    class Config:
        arbitrary_types_allowed = True


class BaseElementNodeParser(NodeParser):
    """
    Splits a document into Text Nodes and Index Nodes corresponding to embedded objects.

    Supports text and tables currently.
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

    @classmethod
    def class_name(cls) -> str:
        return "BaseStructuredNodeParser"

    @classmethod
    def from_defaults(
        cls,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> "BaseElementNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        return cls(
            callback_manager=callback_manager,
            **kwargs,
        )

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            nodes = self.get_nodes_from_node(node)
            all_nodes.extend(nodes)

        return all_nodes

    @abstractmethod
    def get_nodes_from_node(self, node: TextNode) -> List[BaseNode]:
        """Get nodes from node."""

    @abstractmethod
    def extract_elements(self, text: str, **kwargs: Any) -> List[Element]:
        """Extract elements from text."""

    def get_table_elements(self, elements: List[Element]) -> List[Element]:
        """Get table elements."""
        return [e for e in elements if e.type == "table"]

    def get_text_elements(self, elements: List[Element]) -> List[Element]:
        """Get text elements."""
        # TODO: There we should maybe do something with titles
        # and other elements in the future?
        return [e for e in elements if e.type != "table"]

    def extract_table_summaries(self, elements: List[Element]) -> None:
        """Go through elements, extract out summaries that are tables."""
        from llama_index.indices.list.base import SummaryIndex
        from llama_index.service_context import ServiceContext

        llm = self.llm or OpenAI()
        llm = cast(LLM, llm)

        service_context = ServiceContext.from_defaults(llm=llm, embed_model=None)
        for element in tqdm(elements):
            if element.type != "table":
                continue
            index = SummaryIndex.from_documents(
                [Document(text=str(element.element))], service_context=service_context
            )
            query_engine = index.as_query_engine(output_cls=TableOutput)
            try:
                response = query_engine.query(self.summary_query_str)
                element.table_output = cast(PydanticResponse, response).response
            except ValidationError:
                # There was a pydantic validation error, so we will run with text completion
                # fill in the summary and leave other fields blank
                query_engine = index.as_query_engine()
                response_txt = str(query_engine.query(self.summary_query_str))
                element.table_output = TableOutput(summary=response_txt, columns=[])

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

    def get_nodes_and_objects(
        self, nodes: List[BaseNode]
    ) -> Tuple[List[BaseNode], List[IndexNode]]:
        base_nodes, node_mappings = self.get_base_nodes_and_mappings(nodes)

        nodes = []
        objects = []
        for node in base_nodes:
            if isinstance(node, IndexNode):
                node.obj = node_mappings[node.index_id]
                objects.append(node)
            else:
                nodes.append(node)

        return nodes, objects

    def _get_nodes_from_buffer(
        self, buffer: List[str], node_parser: NodeParser
    ) -> List[BaseNode]:
        """Get nodes from buffer."""
        doc = Document(text="\n\n".join(list(buffer)))
        return node_parser.get_nodes_from_documents([doc])

    def get_nodes_from_elements(self, elements: List[Element]) -> List[BaseNode]:
        """Get nodes and mappings."""
        from llama_index.node_parser import SentenceSplitter

        node_parser = SentenceSplitter()

        nodes = []
        cur_text_el_buffer: List[str] = []

        for element in elements:
            if element.type == "table":
                # flush text buffer
                if len(cur_text_el_buffer) > 0:
                    cur_text_nodes = self._get_nodes_from_buffer(
                        cur_text_el_buffer, node_parser
                    )
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
            cur_text_nodes = self._get_nodes_from_buffer(
                cur_text_el_buffer, node_parser
            )
            nodes.extend(cur_text_nodes)
            cur_text_el_buffer = []

        return nodes
