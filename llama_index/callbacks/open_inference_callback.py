"""
Callback handler for logging data to OpenInference files.
"""

import hashlib
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Generic, List, Optional, Sequence, TypeAlias, TypeVar

import pyarrow
from pyarrow.fs import FileSystem

from llama_index.callbacks.base import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType, EventPayload

Embedding: TypeAlias = List[float]

PYARROW_TYPE = "pyarrow_type"
PYARROW_STRING_TYPE = pyarrow.string()
PYARROW_EMBEDDING_TYPE = pyarrow.list_(pyarrow.float64())
PYARROW_LIST_OF_STRINGS_TYPE = pyarrow.list_(pyarrow.string())
PYARROW_LIST_OF_FLOATS_TYPE = pyarrow.list_(pyarrow.float64())


@dataclass
class PyArrowData:
    @classmethod
    def schema(cls) -> pyarrow.Schema:
        return pyarrow.schema(
            [(field_.name, field_.metadata[PYARROW_TYPE]) for field_ in fields(cls)]
        )


PyArrowDataType = TypeVar("PyArrowDataType", bound=PyArrowData)


@dataclass
class QueryData(PyArrowData):
    query_text: Optional[str] = field(
        default=None, metadata={PYARROW_TYPE: PYARROW_STRING_TYPE}
    )
    query_embedding: Optional[Embedding] = field(
        default=None, metadata={PYARROW_TYPE: PYARROW_EMBEDDING_TYPE}
    )
    response_text: Optional[str] = field(
        default=None, metadata={PYARROW_TYPE: PYARROW_STRING_TYPE}
    )
    document_ids: List[str] = field(
        default_factory=list, metadata={PYARROW_TYPE: PYARROW_LIST_OF_STRINGS_TYPE}
    )
    document_hashes: List[str] = field(
        default_factory=list, metadata={PYARROW_TYPE: PYARROW_LIST_OF_STRINGS_TYPE}
    )
    scores: List[float] = field(
        default_factory=list, metadata={PYARROW_TYPE: PYARROW_LIST_OF_FLOATS_TYPE}
    )


@dataclass
class DocumentData(PyArrowData):
    document_text: Optional[str] = field(
        default=None, metadata={PYARROW_TYPE: PYARROW_STRING_TYPE}
    )
    document_embedding: Optional[Embedding] = field(
        default=None, metadata={PYARROW_TYPE: PYARROW_EMBEDDING_TYPE}
    )
    document_id: Optional[str] = field(
        default=None, metadata={PYARROW_TYPE: PYARROW_STRING_TYPE}
    )
    document_hash: Optional[str] = field(
        default=None, metadata={PYARROW_TYPE: PYARROW_STRING_TYPE}
    )


def _generate_id() -> str:
    return hashlib.sha256().hexdigest()


@dataclass
class TraceData:
    trace_id: Optional[str] = field(default_factory=_generate_id)
    query_data: QueryData = field(default_factory=QueryData)
    document_datas: List[DocumentData] = field(default_factory=list)


class PyArrowDataBuffer(Generic[PyArrowDataType]):
    def __init__(self) -> None:
        self._datas: List[PyArrowDataType] = []
        self._size_in_bytes: int = 0

    def append(self, data: PyArrowDataType) -> None:
        self._size_in_bytes += sys.getsizeof(data)
        self._datas.append(data)

    def extend(self, datas: Sequence[PyArrowDataType]) -> None:
        for data in datas:
            self.append(data)

    def to_batch(self) -> pyarrow.RecordBatch:
        return pyarrow.RecordBatch.from_pylist([asdict(data) for data in self._datas])

    def clear(self) -> None:
        self._datas = []
        self._size_in_bytes = 0

    @property
    def size_in_bytes(self) -> int:
        return self._size_in_bytes


def _write_buffer_to_file_system(
    buffer: PyArrowDataBuffer[PyArrowDataType],
    file_system: FileSystem,
    data_path: str,
) -> None:
    batch = buffer.to_batch()
    is_local_file_system = isinstance(file_system, pyarrow.fs.LocalFileSystem)
    if is_local_file_system:
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with file_system.open_output_stream(data_path) as sink:
        writer = pyarrow.ipc.new_stream(sink, batch.schema)
        writer.write_batch(batch)
        writer.close()


class PyArrowLogger(Generic[PyArrowDataType]):
    def __init__(
        self,
        file_system: FileSystem,
        data_path: str,
        max_buffer_size_in_bytes: int,
    ) -> None:
        self._file_system = file_system
        self._data_path = data_path
        self._max_buffer_size_in_bytes = max_buffer_size_in_bytes
        self._buffer = PyArrowDataBuffer[PyArrowDataType]()

    def log_datas(self, datas: List[PyArrowDataType]) -> None:
        self._buffer.extend(datas)
        buffer_full = self._buffer.size_in_bytes >= self._max_buffer_size_in_bytes
        if buffer_full:
            self.flush()

    def flush(self) -> None:
        _write_buffer_to_file_system(
            buffer=self._buffer,
            file_system=self._file_system,
            data_path=self._data_path,
        )
        self._buffer.clear()


class OpenInferenceCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        file_system: FileSystem,
        data_path: str,  # TODO: generalize types
        max_buffer_size_in_bytes: int = 10 * pow(10, 3),
        max_file_size_in_bytes: int = 50 * pow(10, 3),
    ) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self._trace_data = TraceData()
        self._query_data_logger = PyArrowLogger[QueryData](
            file_system=file_system,
            data_path=os.path.join(data_path, "query.arrow"),
            max_buffer_size_in_bytes=max_buffer_size_in_bytes,
        )
        self._document_data_logger = PyArrowLogger[DocumentData](
            file_system=file_system,
            data_path=os.path.join(data_path, "document.arrow"),
            max_buffer_size_in_bytes=max_buffer_size_in_bytes,
        )

    @classmethod
    def to_local_file_system(cls, data_path: str) -> "OpenInferenceCallbackHandler":
        return cls(pyarrow.fs.LocalFileSystem(), data_path)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""
        if trace_id == "query":
            self._trace_data = TraceData()

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        if trace_id == "query":
            self._query_data_logger.log_datas([self._trace_data.query_data])
            self._document_data_logger.log_datas(self._trace_data.document_datas)

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Run when an event starts and return id of event."""
        if payload is not None:
            if event_type is CBEventType.QUERY:
                self._trace_data.query_data.query_text = payload[EventPayload.QUERY_STR]
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when an event ends."""
        if payload is not None:
            if event_type is CBEventType.RETRIEVE:
                for node_with_score in payload[EventPayload.NODES]:
                    node = node_with_score.node
                    score = node_with_score.score
                    self._trace_data.query_data.document_hashes.append(node.hash)
                    self._trace_data.query_data.document_ids.append(node.id_)
                    self._trace_data.query_data.scores.append(score)
                    self._trace_data.document_datas.append(
                        DocumentData(
                            document_text=node.text,
                            document_id=node.id_,
                            document_hash=node.hash,
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

    def flush_log_buffers(self) -> None:
        self._query_data_logger.flush()
        self._document_data_logger.flush()
