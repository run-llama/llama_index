"""
Callback handler for storing generation data in OpenInference format.
OpenInference is an open standard for capturing and storing AI model inferences.
It enables production LLMapp servers to seamlessly integrate with LLM
observability solutions such as Arize and Phoenix.

For more information on the specification, see
https://github.com/Arize-ai/open-inference-spec
"""

import importlib
import uuid
from dataclasses import dataclass, field, fields
from datetime import datetime
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
)

from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload

if TYPE_CHECKING:
    from pandas import DataFrame


OPENINFERENCE_COLUMN_NAME = "openinference_column_name"
Embedding = List[float]


def _generate_random_id() -> str:
    """
    Generates a random ID.

    Returns:
        str: A random ID.

    """
    return str(uuid.uuid4())


@dataclass
class QueryData:
    """
    Query data with column names following the OpenInference specification.
    """

    id: str = field(
        default_factory=_generate_random_id,
        metadata={OPENINFERENCE_COLUMN_NAME: ":id.id:"},
    )
    timestamp: Optional[str] = field(
        default=None, metadata={OPENINFERENCE_COLUMN_NAME: ":timestamp.iso_8601:"}
    )
    query_text: Optional[str] = field(
        default=None,
        metadata={OPENINFERENCE_COLUMN_NAME: ":feature.text:prompt"},
    )
    query_embedding: Optional[Embedding] = field(
        default=None,
        metadata={OPENINFERENCE_COLUMN_NAME: ":feature.[float].embedding:prompt"},
    )
    llm_prompt: Optional[str] = field(
        default=None,
        metadata={OPENINFERENCE_COLUMN_NAME: ":feature.text:llm_prompt"},
    )
    llm_messages: Optional[Tuple[str, str]] = field(
        default=None,
        metadata={OPENINFERENCE_COLUMN_NAME: ":feature.[str]:llm_messages"},
    )
    response_text: Optional[str] = field(
        default=None, metadata={OPENINFERENCE_COLUMN_NAME: ":prediction.text:response"}
    )
    node_ids: List[str] = field(
        default_factory=list,
        metadata={
            OPENINFERENCE_COLUMN_NAME: ":feature.[str].retrieved_document_ids:prompt"
        },
    )
    scores: List[float] = field(
        default_factory=list,
        metadata={
            OPENINFERENCE_COLUMN_NAME: (
                ":feature.[float].retrieved_document_scores:prompt"
            )
        },
    )


@dataclass
class NodeData:
    """Node data."""

    id: str
    node_text: Optional[str] = None
    node_embedding: Optional[Embedding] = None


BaseDataType = TypeVar("BaseDataType", QueryData, NodeData)


def as_dataframe(data: Iterable[BaseDataType]) -> "DataFrame":
    """
    Converts a list of BaseDataType to a pandas dataframe.

    Args:
        data (Iterable[BaseDataType]): A list of BaseDataType.

    Returns:
        DataFrame: The converted pandas dataframe.

    """
    pandas = _import_package("pandas")
    as_dict_list = []
    for datum in data:
        as_dict = {
            field.metadata.get(OPENINFERENCE_COLUMN_NAME, field.name): getattr(
                datum, field.name
            )
            for field in fields(datum)
        }
        as_dict_list.append(as_dict)

    return pandas.DataFrame(as_dict_list)


@dataclass
class TraceData:
    """Trace data."""

    query_data: QueryData = field(default_factory=QueryData)
    node_datas: List[NodeData] = field(default_factory=list)


def _import_package(package_name: str) -> ModuleType:
    """
    Dynamically imports a package.

    Args:
        package_name (str): Name of the package to import.

    Raises:
        ImportError: If the package is not installed.

    Returns:
        ModuleType: The imported package.

    """
    try:
        package = importlib.import_module(package_name)
    except ImportError:
        raise ImportError(f"The {package_name} package must be installed.")
    return package


class OpenInferenceCallbackHandler(BaseCallbackHandler):
    """
    Callback handler for storing generation data in OpenInference format.
    OpenInference is an open standard for capturing and storing AI model
    inferences. It enables production LLMapp servers to seamlessly integrate
    with LLM observability solutions such as Arize and Phoenix.

    For more information on the specification, see
    https://github.com/Arize-ai/open-inference-spec
    """

    def __init__(
        self,
        callback: Optional[Callable[[List[QueryData], List[NodeData]], None]] = None,
    ) -> None:
        """
        Initializes the OpenInferenceCallbackHandler.

        Args:
            callback (Optional[Callable[[List[QueryData], List[NodeData]], None]], optional): A
            callback function that will be called when a query trace is
            completed, often used for logging or persisting query data.

        """
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self._callback = callback
        self._trace_data = TraceData()
        self._query_data_buffer: List[QueryData] = []
        self._node_data_buffer: List[NodeData] = []

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        if trace_id == "query" or trace_id == "chat":
            self._trace_data = TraceData()
            self._trace_data.query_data.timestamp = datetime.now().isoformat()
            self._trace_data.query_data.id = _generate_random_id()

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        if trace_id == "query" or trace_id == "chat":
            self._query_data_buffer.append(self._trace_data.query_data)
            self._node_data_buffer.extend(self._trace_data.node_datas)
            self._trace_data = TraceData()
            if self._callback is not None:
                self._callback(self._query_data_buffer, self._node_data_buffer)

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        if payload is not None:
            if event_type is CBEventType.QUERY:
                query_text = payload[EventPayload.QUERY_STR]
                self._trace_data.query_data.query_text = query_text
            elif event_type is CBEventType.LLM:
                if prompt := payload.get(EventPayload.PROMPT, None):
                    self._trace_data.query_data.llm_prompt = prompt
                if messages := payload.get(EventPayload.MESSAGES, None):
                    self._trace_data.query_data.llm_messages = [
                        (m.role.value, m.content) for m in messages
                    ]
                    # For chat engines there is no query event and thus the
                    # query text will be None, in this case we set the query
                    # text to the last message passed to the LLM
                    if self._trace_data.query_data.query_text is None:
                        self._trace_data.query_data.query_text = messages[-1].content
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if payload is None:
            return
        if event_type is CBEventType.RETRIEVE:
            for node_with_score in payload[EventPayload.NODES]:
                node = node_with_score.node
                score = node_with_score.score
                self._trace_data.query_data.node_ids.append(node.hash)
                self._trace_data.query_data.scores.append(score)
                self._trace_data.node_datas.append(
                    NodeData(
                        id=node.hash,
                        node_text=node.text,
                    )
                )
        elif event_type is CBEventType.LLM:
            if self._trace_data.query_data.response_text is None:
                if response := payload.get(EventPayload.RESPONSE, None):
                    if isinstance(response, ChatResponse):
                        # If the response is of class ChatResponse the string
                        # representation has the format "<role>: <message>",
                        # but we want just the message
                        response_text = response.message.content
                    else:
                        response_text = str(response)
                    self._trace_data.query_data.response_text = response_text
                elif completion := payload.get(EventPayload.COMPLETION, None):
                    self._trace_data.query_data.response_text = str(completion)
        elif event_type is CBEventType.EMBEDDING:
            self._trace_data.query_data.query_embedding = payload[
                EventPayload.EMBEDDINGS
            ][0]

    def flush_query_data_buffer(self) -> List[QueryData]:
        """
        Clears the query data buffer and returns the data.

        Returns:
            List[QueryData]: The query data.

        """
        query_data_buffer = self._query_data_buffer
        self._query_data_buffer = []
        return query_data_buffer

    def flush_node_data_buffer(self) -> List[NodeData]:
        """
        Clears the node data buffer and returns the data.

        Returns:
            List[NodeData]: The node data.

        """
        node_data_buffer = self._node_data_buffer
        self._node_data_buffer = []
        return node_data_buffer
