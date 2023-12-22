from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import pandas as pd
from tqdm import tqdm

from llama_index.bridge.pydantic import BaseModel, Field, ValidationError
from llama_index.callbacks.base import CallbackManager
from llama_index.llms.llm import LLM
from llama_index.llms.openai import OpenAI
from llama_index.node_parser.interface import NodeParser
from llama_index.response.schema import PydanticResponse
from llama_index.schema import BaseNode, Document, IndexNode, MetadataMode, TextNode
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
    title_level: Optional[int] = None
    table_output: Optional[TableOutput] = None
    table: Optional[pd.DataFrame] = None

    class Config:
        arbitrary_types_allowed = True


def md_to_df(md_str: str) -> pd.DataFrame:
    """Convert Markdown to dataframe."""
    # Replace markdown pipe tables with commas
    md_str = md_str.replace("|", ",")

    # Remove the second line (table header separator)
    lines = md_str.split("\n")
    md_str = "\n".join(lines[:1] + lines[2:])

    # Remove the first and last character (the pipes)
    lines = md_str.split("\n")
    md_str = "\n".join([line[1:-1] for line in lines])

    """ Check if the table is empty"""
    if len(md_str) == 0:
        return None

    # Use pandas to read the CSV string into a DataFrame
    return pd.read_csv(StringIO(md_str))


def filter_table(table_element: Any) -> bool:
    """Filter table."""
    table_df = md_to_df(table_element.element)
    """ check if table_df is not None, has more than one row, and more than one column """
    return table_df is not None and not table_df.empty and len(table_df.columns) > 1


def extract_markdown_elements(
    markdown: str, table_filters: Optional[List[Callable]] = None
) -> List[Element]:
    lines = markdown.split("\n")
    currentElement = None

    elements: List[Element] = []
    """ Then parse the lines """
    for line in lines:
        if line.startswith("```"):
            """check if this is the end of a code block"""
            if currentElement is not None and currentElement.type == "code":
                elements.append(currentElement)
                currentElement = None
                """ if there is some text after the ``` create a text element with it"""
                if len(line) > 3:
                    elements.append(
                        Element(
                            id=f"id_{len(elements)}",
                            type="text",
                            element=line.lstrip("```"),
                        )
                    )

            elif line.count("```") == 2 and line[-3] != "`":
                """check if inline code block (aka have a second ``` in line but not at the end)"""
                if currentElement is not None:
                    elements.append(currentElement)
                currentElement = Element(
                    id=f"id_{len(elements)}", type="code", element=line.lstrip("```")
                )
            elif currentElement is not None and currentElement.type == "text":
                currentElement.element += "\n" + line
            else:
                if currentElement is not None:
                    elements.append(currentElement)
                currentElement = Element(
                    id=f"id_{len(elements)}", type="text", element=line
                )

        elif currentElement is not None and currentElement.type == "code":
            currentElement.element += "\n" + line

        elif line.startswith("|"):
            if currentElement is not None and currentElement.type != "table":
                if currentElement is not None:
                    elements.append(currentElement)
                currentElement = Element(
                    id=f"id_{len(elements)}", type="table", element=line
                )
            elif currentElement is not None:
                currentElement.element += "\n" + line
            else:
                currentElement = Element(
                    id=f"id_{len(elements)}", type="table", element=line
                )
        elif line.startswith("#"):
            if currentElement is not None:
                elements.append(currentElement)
            currentElement = Element(
                id=f"id_{len(elements)}",
                type="title",
                element=line.lstrip("#"),
                title_level=len(line) - len(line.lstrip("#")),
            )
        else:
            if currentElement is not None and currentElement.type != "text":
                elements.append(currentElement)
                currentElement = Element(
                    id=f"id_{len(elements)}", type="text", element=line
                )
            elif currentElement is not None:
                currentElement.element += "\n" + line
            else:
                currentElement = Element(
                    id=f"id_{len(elements)}", type="text", element=line
                )
    if currentElement is not None:
        elements.append(currentElement)

    """for each elements"""
    for idx, element in enumerate(elements):
        if element.type == "table":
            if table_filters is not None:
                should_keep = all(tf(element) for tf in table_filters)
            else:
                should_keep = True  # default to keeping all tables
            """ if the element is a table, convert it to a dataframe"""
            if should_keep:
                table = md_to_df(element.element)
                elements[idx] = Element(
                    id=f"id_{idx}", type="table", element=element, table=table
                )
            else:
                elements[idx] = Element(
                    id=f"id_{idx}",
                    type="text",
                    element=element.element,
                )
        else:
            """if the element is not a table, keep it as to text"""
            elements[idx] = Element(
                id=f"id_{idx}",
                type="text",
                element=element.element,
            )

    # merge consecutive text elements together for now
    merged_elements: List[Element] = []
    for element in elements:
        if (
            len(merged_elements) > 0
            and element.type == "text"
            and merged_elements[-1].type == "text"
        ):
            merged_elements[-1].element += "\n" + element.element
        else:
            merged_elements.append(element)
    elements = merged_elements
    return merged_elements


def extract_table_summaries(
    elements: List[Element], llm: Optional[Any], summary_query_str: str
) -> None:
    """Go through elements, extract out summaries that are tables."""
    from llama_index.indices.list.base import SummaryIndex
    from llama_index.service_context import ServiceContext

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
    # There we should maybe do something with titles and other elements in the future?
    return [e for e in elements if e.type != "table"]


def _get_nodes_from_buffer(
    buffer: List[str], node_parser: NodeParser
) -> List[BaseNode]:
    """Get nodes from buffer."""
    doc = Document(text="\n\n".join(list(buffer)))
    return node_parser.get_nodes_from_documents([doc])


def get_nodes_from_elements(elements: List[Element]) -> List[BaseNode]:
    """Get nodes and mappings."""
    from llama_index.node_parser import SentenceSplitter

    node_parser = SentenceSplitter()

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


class MarkdownElementNodeParser(NodeParser):
    """Markdown element node parser.

    Splits a markdown document into Text Nodes and Index Nodes corresponding to embedded objects
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
        callback_manager = callback_manager or CallbackManager([])

        return super().__init__(
            callback_manager=callback_manager,
            llm=llm,
            summary_query_str=summary_query_str,
        )

    @classmethod
    def class_name(cls) -> str:
        return "MarkdownElementNodeParser"

    @classmethod
    def from_defaults(
        cls,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "MarkdownElementNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        return cls(
            callback_manager=callback_manager,
        )

    def get_nodes_from_node(self, node: TextNode) -> List[BaseNode]:
        """Get nodes from node."""
        elements = extract_markdown_elements(
            node.get_content(metadata_mode=MetadataMode.NONE),
            table_filters=[filter_table],
        )
        table_elements = get_table_elements(elements)
        # extract summaries over table elements
        extract_table_summaries(table_elements, self.llm, self.summary_query_str)
        # convert into nodes
        # will return a list of Nodes and Index Nodes
        return get_nodes_from_elements(elements)

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
