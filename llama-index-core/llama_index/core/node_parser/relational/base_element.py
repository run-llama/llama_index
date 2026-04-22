import uuid
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast
from tqdm import tqdm

from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs, asyncio_run
from llama_index.core.base.response.schema import PydanticResponse
from llama_index.core.bridge.pydantic import (
    BaseModel,
    Field,
    ValidationError,
    ConfigDict,
)
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import (
    BaseNode,
    Document,
    IndexNode,
    MetadataMode,
    TextNode,
)
from llama_index.core.utils import get_tqdm_iterable

DEFAULT_SUMMARY_QUERY_STR = """\
What is this table about? Give a very concise summary (imagine you are adding a new caption and summary for this table), \
and output the real/existing table title/caption if context provided.\
and output the real/existing table id if context provided.\
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
    table_title: Optional[str] = None
    table_id: Optional[str] = None
    columns: List[TableColumnOutput]


class Element(BaseModel):
    """Element object."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str
    type: str
    element: Any
    title_level: Optional[int] = None
    table_output: Optional[TableOutput] = None
    table: Optional[Any] = None
    markdown: Optional[str] = None
    page_number: Optional[int] = None


class BaseElementNodeParser(NodeParser):
    """
    Splits a document into Text Nodes and Index Nodes corresponding to embedded objects.

    Supports text and tables currently.
    """

    callback_manager: CallbackManager = Field(
        default_factory=lambda: CallbackManager([]), exclude=True
    )
    llm: Optional[LLM] = Field(
        default=None, description="LLM model to use for summarization."
    )
    summary_query_str: str = Field(
        default=DEFAULT_SUMMARY_QUERY_STR,
        description="Query string to use for summarization.",
    )
    num_workers: int = Field(
        default=DEFAULT_NUM_WORKERS,
        description="Num of workers for async jobs.",
    )

    show_progress: bool = Field(default=True, description="Whether to show progress.")

    nested_node_parser: Optional[NodeParser] = Field(
        default=None,
        description="Other types of node parsers to handle some types of nodes.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "BaseElementNodeParser"

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

    async def _aparse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            nodes = await self.aget_nodes_from_node(node)
            all_nodes.extend(nodes)

        return all_nodes

    @abstractmethod
    def get_nodes_from_node(self, node: TextNode) -> List[BaseNode]:
        """Get nodes from node."""
        ...

    async def aget_nodes_from_node(self, node: TextNode) -> List[BaseNode]:
        """Get nodes from node."""
        return self.get_nodes_from_node(node)

    @abstractmethod
    def extract_elements(self, text: str, **kwargs: Any) -> List[Element]:
        """Extract elements from text."""

    def get_table_elements(self, elements: List[Element]) -> List[Element]:
        """Get table elements."""
        return [e for e in elements if e.type == "table" or e.type == "table_text"]

    def get_text_elements(self, elements: List[Element]) -> List[Element]:
        """Get text elements."""
        # TODO: There we should maybe do something with titles
        # and other elements in the future?
        return [e for e in elements if e.type != "table"]

    def _get_context_lines(
        self, element_text: str, from_end: bool = False, num_lines: int = 3
    ) -> Optional[str]:
        """Extract N lines from element if any starts with 'table'."""
        lines = element_text.strip().split("\n")
        if from_end:
            # Get last N lines (for previous element, we want lines near the table)
            context_lines = lines[-num_lines:] if len(lines) >= num_lines else lines
        else:
            # Get first N lines (for next element)
            context_lines = lines[:num_lines] if len(lines) >= num_lines else lines

        # Check if any of the context lines starts with "table"
        has_table_context = any(
            line.lower().strip().startswith("table") for line in context_lines
        )

        if has_table_context:
            return "\n".join(context_lines)
        return None

    def extract_table_summaries(self, elements: List[Element]) -> None:
        """Go through all elements, extract out summaries that are tables."""
        from llama_index.core.indices.list.base import SummaryIndex
        from llama_index.core.settings import Settings

        llm = self.llm or Settings.llm

        table_context_list = []
        for idx, element in tqdm(enumerate(elements)):
            if element.type not in ("table", "table_text"):
                continue
            table_context = str(element.element)

            # Check previous element - get last 3 lines if any of them start with "table"
            if idx > 0 and elements[idx - 1].type == "text":
                prev_context = self._get_context_lines(
                    str(elements[idx - 1].element), from_end=True, num_lines=3
                )
                if prev_context:
                    table_context = prev_context + "\n" + table_context

            # Check next element - get first 3 lines if any of them start with "table"
            if idx < len(elements) - 1 and elements[idx + 1].type == "text":
                next_context = self._get_context_lines(
                    str(elements[idx + 1].element), from_end=False, num_lines=3
                )
                if next_context:
                    table_context += "\n" + next_context

            table_context_list.append(table_context)

        async def _get_table_output(table_context: str, summary_query_str: str) -> Any:
            index = SummaryIndex.from_documents(
                [Document(text=table_context)],
            )
            query_engine = index.as_query_engine(llm=llm, output_cls=TableOutput)
            try:
                response = await query_engine.aquery(summary_query_str)
                if isinstance(response, PydanticResponse):
                    return response.response
                else:
                    raise ValueError(f"Expected PydanticResponse, got {type(response)}")
            except (ValidationError, ValueError):
                # There was a pydantic validation error, so we will run with text completion
                # fill in the summary and leave other fields blank
                query_engine = index.as_query_engine(llm=llm)
                response_txt = await query_engine.aquery(summary_query_str)
                return TableOutput(summary=str(response_txt), columns=[])

        summary_jobs = [
            _get_table_output(table_context, self.summary_query_str)
            for table_context in table_context_list
        ]
        summary_co = run_jobs(summary_jobs, workers=self.num_workers)
        summary_outputs = asyncio_run(summary_co)
        # Only assign table_output to table elements (not all elements)
        table_elements = [e for e in elements if e.type in ("table", "table_text")]
        for element, summary_output in zip(table_elements, summary_outputs):
            element.table_output = summary_output

    async def aextract_table_summaries(self, elements: List[Element]) -> None:
        """Go through elements, extract out summaries that are tables."""
        from llama_index.core.indices.list.base import SummaryIndex
        from llama_index.core.settings import Settings

        llm = self.llm or Settings.llm

        table_context_list = []
        if elements:
            for idx, element in tqdm(enumerate(elements)):
                if element.type not in ("table", "table_text"):
                    continue
                table_context = str(element.element)

                # Check previous element - get last 3 lines if any of them start with "table"
                if idx > 0 and elements[idx - 1].type == "text":
                    prev_context = self._get_context_lines(
                        str(elements[idx - 1].element), from_end=True, num_lines=3
                    )
                    if prev_context:
                        table_context = prev_context + "\n" + table_context

                # Check next element - get first 3 lines if any of them start with "table"
                if idx < len(elements) - 1 and elements[idx + 1].type == "text":
                    next_context = self._get_context_lines(
                        str(elements[idx + 1].element), from_end=False, num_lines=3
                    )
                    if next_context:
                        table_context += "\n" + next_context

                table_context_list.append(table_context)

        async def _get_table_output(table_context: str, summary_query_str: str) -> Any:
            index = SummaryIndex.from_documents(
                [Document(text=table_context)],
            )
            query_engine = index.as_query_engine(llm=llm, output_cls=TableOutput)
            try:
                response = await query_engine.aquery(summary_query_str)
                return cast(PydanticResponse, response).response
            except (ValidationError, ValueError):
                # There was a pydantic validation error, so we will run with text completion
                # fill in the summary and leave other fields blank
                query_engine = index.as_query_engine(llm=llm)
                response_txt = await query_engine.aquery(summary_query_str)
                return TableOutput(summary=str(response_txt), columns=[])

        summary_jobs = [
            _get_table_output(table_context, self.summary_query_str)
            for table_context in table_context_list
        ]
        summary_outputs = await run_jobs(summary_jobs, workers=self.num_workers)
        # Only assign table_output to table elements (not all elements)
        table_elements = [e for e in elements if e.type in ("table", "table_text")]
        for element, summary_output in zip(table_elements, summary_outputs):
            element.table_output = summary_output

    def get_base_nodes_and_mappings(
        self, nodes: List[BaseNode]
    ) -> Tuple[List[BaseNode], Dict]:
        """
        Get base nodes and mappings.

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

    def get_nodes_from_elements(
        self,
        elements: List[Element],
        node_inherited: Optional[TextNode] = None,
        ref_doc_text: Optional[str] = None,
    ) -> List[BaseNode]:
        """Get nodes and mappings."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for this function. Please install it with `pip install pandas`."
            )

        from llama_index.core.node_parser import SentenceSplitter

        node_parser = self.nested_node_parser or SentenceSplitter()

        nodes: List[BaseNode] = []
        cur_text_el_buffer: List[str] = []
        for element in elements:
            if element.type == "table" or element.type == "table_text":
                # flush text buffer for table
                if len(cur_text_el_buffer) > 0:
                    cur_text_nodes = self._get_nodes_from_buffer(
                        cur_text_el_buffer, node_parser
                    )
                    nodes.extend(cur_text_nodes)
                    cur_text_el_buffer = []

                table_output = cast(TableOutput, element.table_output)
                table_md = ""
                if element.type == "table":
                    table_df = cast(pd.DataFrame, element.table)
                    # We serialize the table as markdown as it allow better accuracy
                    # We do not use the table_df.to_markdown() method as it generate
                    # a table with a token hungry format.
                    table_md = "|"
                    for col_name, col in table_df.items():
                        table_md += f"{col_name}|"
                    table_md += "\n|"
                    for col_name, col in table_df.items():
                        table_md += f"---|"
                    table_md += "\n"
                    for row in table_df.itertuples():
                        table_md += "|"
                        for col in row[1:]:
                            table_md += f"{col}|"
                        table_md += "\n"
                elif element.type == "table_text":
                    # if the table is non-perfect table, we still want to keep the original text of table
                    table_md = str(element.element)

                col_schema = "\n\n".join([str(col) for col in table_output.columns])

                # We build a summary of the table containing the extracted summary, and a description of the columns
                table_summary = str(table_output.summary)
                if table_output.table_title:
                    table_summary += ",\nwith the following table title:\n"
                    table_summary += str(table_output.table_title)

                table_summary += ",\nwith the following columns:\n"

                for col in table_output.columns:
                    table_summary += f"- {col.col_name}: {col.summary}\n"

                # attempt to find start_char_idx for table
                # raw table string regardless if perfect or not is stored in element.element

                if ref_doc_text:
                    start_char_idx = ref_doc_text.find(str(element.element))
                    if start_char_idx >= 0:
                        end_char_idx = start_char_idx + len(str(element.element))
                    else:
                        start_char_idx = None  # type: ignore
                        end_char_idx = None  # type: ignore
                else:
                    start_char_idx = None  # type: ignore
                    end_char_idx = None  # type: ignore

                # shared index_id and node_id
                node_id = str(uuid.uuid4())
                index_node = IndexNode(
                    text=table_summary,
                    metadata={
                        "col_schema": col_schema,
                    },
                    excluded_embed_metadata_keys=["col_schema"],
                    index_id=node_id,
                    start_char_idx=start_char_idx,
                    end_char_idx=end_char_idx,
                )

                table_str = table_summary + "\n" + table_md

                text_node = TextNode(
                    id_=node_id,
                    text=table_str,
                    metadata={
                        # serialize the table as a dictionary string for dataframe of perfect table
                        "table_df": (
                            str(table_df.to_dict())
                            if element.type == "table"
                            else table_md
                        ),
                        # add table summary for retrieval purposes
                        "table_summary": table_summary,
                    },
                    excluded_embed_metadata_keys=["table_df", "table_summary"],
                    excluded_llm_metadata_keys=["table_df", "table_summary"],
                    start_char_idx=start_char_idx,
                    end_char_idx=end_char_idx,
                )
                nodes.extend([index_node, text_node])
            else:
                cur_text_el_buffer.append(str(element.element))

        # flush text buffer for the last batch
        if len(cur_text_el_buffer) > 0:
            cur_text_nodes = self._get_nodes_from_buffer(
                cur_text_el_buffer, node_parser
            )
            nodes.extend(cur_text_nodes)
            cur_text_el_buffer = []

        # remove empty nodes and keep node original metadata inherited from parent nodes
        for node in nodes:
            if node_inherited and node_inherited.metadata:
                node.metadata.update(node_inherited.metadata)
                node.excluded_embed_metadata_keys = (
                    node_inherited.excluded_embed_metadata_keys
                )
                node.excluded_llm_metadata_keys = (
                    node_inherited.excluded_llm_metadata_keys
                )
        return [
            node
            for node in nodes
            if len(node.get_content(metadata_mode=MetadataMode.NONE)) > 0
        ]

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[BaseNode]:
        nodes = self.get_nodes_from_documents(nodes, **kwargs)  # type: ignore
        nodes, objects = self.get_nodes_and_objects(nodes)
        return nodes + objects  # type: ignore

    async def acall(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[BaseNode]:
        nodes = await self.aget_nodes_from_documents(nodes, **kwargs)  # type: ignore
        nodes, objects = self.get_nodes_and_objects(nodes)
        return nodes + objects  # type: ignore
