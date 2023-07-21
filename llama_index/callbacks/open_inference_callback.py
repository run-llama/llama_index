"""
Callback handler for retrieval-augmented generation data in OpenInference
format.
"""

import hashlib
import importlib
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from datetime import datetime
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TypeAlias,
    TypeVar,
)

from llama_index.callbacks.base import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType, EventPayload

if TYPE_CHECKING:
    from pandas import DataFrame


OPENINFERENCE_COLUMN_NAME = "openinference_column_name"
BaseDataType = TypeVar("BaseDataType", bound="BaseData")
Embedding: TypeAlias = List[float]


def _hash_bytes_with_sha256(data: bytes) -> str:
    """Hashes a byte string using SHA256.

    Args:
        data (bytes): Data to be hashed in bytes.

    Returns:
        str: A hash of the input as a string.
    """
    return hashlib.sha256(data).hexdigest()


def _generate_random_id() -> str:
    """Generates a random ID.

    Returns:
        str: A random ID.
    """
    return _hash_bytes_with_sha256(os.urandom(32))


@dataclass
class BaseData:
    """A serializable base class for retrieval-augmented data."""

    id: str = field(
        default_factory=_generate_random_id,
        metadata={OPENINFERENCE_COLUMN_NAME: ":id.str:"},
    )

    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass to a dictionary.

        Returns:
            Dict[str, Any]: Representation of the dataclass as a dictionary.
        """
        return {
            field.metadata.get(OPENINFERENCE_COLUMN_NAME, field.name): getattr(
                self, field.name
            )
            for field in fields(self)
        }


@dataclass
class QueryData(BaseData):
    """
    Query data for retrieval-augmented generation data, with column names
    following the OpenInference specification.
    """

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
    response_text: Optional[str] = field(
        default=None, metadata={OPENINFERENCE_COLUMN_NAME: ":prediction.text:response"}
    )
    document_ids: List[str] = field(
        default_factory=list,
        metadata={
            OPENINFERENCE_COLUMN_NAME: ":feature.[str].retrieved_document_ids:prompt"
        },
    )
    scores: List[float] = field(
        default_factory=list,
        metadata={
            OPENINFERENCE_COLUMN_NAME: ":feature.[float].retrieved_document_scores:prompt"
        },
    )


@dataclass
class DocumentData(BaseData):
    """Document data for retrieval-augmented generation."""

    document_text: Optional[str] = None
    document_embedding: Optional[Embedding] = None


@dataclass
class TraceData:
    """Trace data"""

    query_data: QueryData = field(default_factory=QueryData)
    document_datas: List[DocumentData] = field(default_factory=list)


def _import_package(package_name: str) -> ModuleType:
    """Dynamically imports a package.

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


class DataBuffer(Generic[BaseDataType]):
    def __init__(self, max_size_in_bytes: Optional[int] = None) -> None:
        """The data buffer for OpenInference data.

        Args:
            max_size_in_bytes (Optional[int], optional): The maximum size of the
                buffer in bytes. Once this size is reached, the oldest data will
                be removed until the size of the buffer is less than the maximum
                size.
        """
        self._id_to_data_map: OrderedDict[str, BaseData] = OrderedDict()
        self._max_size_in_bytes = max_size_in_bytes
        self._size_in_bytes = 0

    def __len__(self) -> int:
        return len(self._id_to_data_map)

    def add(self, data: BaseData) -> None:
        """Add data to the buffer. If the buffer is full, the oldest data will
        be removed until the size of the buffer is less than the maximum
        configured size.

        Args:
            data (BaseData): Data to add to the buffer.
        """
        if data.id in self._id_to_data_map:
            self._size_in_bytes -= sys.getsizeof(self._id_to_data_map.pop(data.id))
        self._id_to_data_map[data.id] = data
        self._size_in_bytes += sys.getsizeof(data)

        if self._max_size_in_bytes is not None:
            while self._size_in_bytes > self._max_size_in_bytes:
                _, data = self._id_to_data_map.popitem(last=False)
                self._size_in_bytes -= sys.getsizeof(data)

    def clear(self) -> None:
        """Clears the buffer."""
        self._id_to_data_map.clear()
        self._size_in_bytes = 0

    @property
    def dataframe(self) -> "DataFrame":
        """Returns the data buffer as a pandas dataframe.

        Returns:
            DataFrame: The buffer dataframe.
        """
        pandas = _import_package("pandas")
        return pandas.DataFrame(
            [data.to_dict() for data in self._id_to_data_map.values()]
        )

    @property
    def size_in_bytes(self) -> int:
        """Returns the size of the buffer in bytes.

        Returns:
            int: The size of the buffer in bytes.
        """
        return self._size_in_bytes


class OpenInferenceCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        max_size_in_bytes: Optional[int] = None,
        logger: Optional[Callable[[DataBuffer[QueryData]], None]] = None,
    ) -> None:
        """Initializer for the OpenInferenceCallbackHandler.

        Args:
            max_size_in_bytes (Optional[int], optional): The buffer size
                threshold. If this argument is set to None, the buffer will not
                be limited in size.

            logger (Optional[Callable[[OpenInferenceDataBuffer[QueryData]],
                None]], optional): A callback function that will be called at
                the end of each trace, typically used to persist data (e.g., to
                disk, cloud storage, or by sending to a data ingestion service).
        """
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self._callback = logger
        self._trace_data = TraceData()
        self._query_data_buffer = DataBuffer[QueryData](
            max_size_in_bytes=max_size_in_bytes
        )
        self._document_data_buffer = DataBuffer[DocumentData](
            max_size_in_bytes=max_size_in_bytes,
        )

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        if trace_id == "query":
            trace_start_time = datetime.now()
            self._trace_data = TraceData()
            self._trace_data.query_data.timestamp = trace_start_time.isoformat()
            self._trace_data.query_data.id = _generate_random_id()

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        if trace_id == "query":
            self._query_data_buffer.add(self._trace_data.query_data)
            for document_data in self._trace_data.document_datas:
                self._document_data_buffer.add(document_data)
            if self._callback is not None:
                self._callback(self._query_data_buffer)

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> str:
        if payload is not None:
            if event_type is CBEventType.QUERY:
                query_text = payload[EventPayload.QUERY_STR]
                self._trace_data.query_data.id = _hash_bytes_with_sha256(
                    query_text.encode()
                )
                self._trace_data.query_data.query_text = query_text
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if payload is not None:
            if event_type is CBEventType.RETRIEVE:
                for node_with_score in payload[EventPayload.NODES]:
                    node = node_with_score.node
                    score = node_with_score.score
                    self._trace_data.query_data.document_ids.append(node.id_)
                    self._trace_data.query_data.scores.append(score)
                    self._trace_data.document_datas.append(
                        DocumentData(
                            id=node.id_,
                            document_text=node.text,
                        )
                    )
            elif event_type is CBEventType.LLM:
                self._trace_data.query_data.response_text = payload[
                    EventPayload.RESPONSE
                ]
            elif event_type is CBEventType.EMBEDDING:
                self._trace_data.query_data.query_embedding = payload[
                    EventPayload.EMBEDDINGS
                ][
                    0
                ]  # when does this list have more than one element?

    @property
    def query_dataframe(self) -> "DataFrame":
        """A pandas dataframe representing the query data.

        Returns:
            DataFrame: The query dataframe.
        """
        return self._query_data_buffer.dataframe
