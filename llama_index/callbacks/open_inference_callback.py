"""
Callback handler for logging data to OpenInference files.
"""

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, TypeAlias

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
class Data:
    def pyarrow_schema(self) -> pyarrow.Schema:
        return pyarrow.schema(
            [(field_.name, field_.metadata[PYARROW_TYPE]) for field_ in fields(self)]
        )


@dataclass
class QueryData(Data):
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
class DocumentData(Data):
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


class OpenInferenceCallbackHandler(BaseCallbackHandler):
    def __init__(
        self, file_system: FileSystem, data_path: str  # TODO: generalize types
    ) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self._query_data = QueryData()
        self._document_datas: List[DocumentData] = []
        self._query_data_buffer: List[QueryData] = []
        self._document_data_buffer: List[DocumentData] = []
        self._file_system = file_system
        self._data_path = data_path
        self._max_batch_size = 2  # TODO: make configurable

    @classmethod
    def to_local_file_system(cls, data_path: str) -> "OpenInferenceCallbackHandler":
        return cls(pyarrow.fs.LocalFileSystem(), data_path)

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
                self._query_data.query_text = payload[EventPayload.QUERY_STR]
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
                    self._query_data.document_hashes.append(node.hash)
                    self._query_data.document_ids.append(node.id_)
                    self._query_data.scores.append(score)
                    self._document_datas.append(
                        DocumentData(
                            document_text=node.text,
                            document_id=node.id_,
                            document_hash=node.hash,
                        )
                    )
            elif event_type is CBEventType.LLM:
                self._query_data.response_text = payload[EventPayload.RESPONSE]
            elif event_type is CBEventType.EMBEDDING:
                self._query_data.query_embedding = payload[EventPayload.EMBEDDINGS][
                    0
                ]  # when does this list have more than one element?

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""
        if trace_id == "query":
            self._query_data = QueryData()

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        if trace_id == "query":
            self._add_to_buffer(self._query_data, self._document_datas)
            if len(self._query_data_buffer) >= self._max_batch_size:
                self._flush_buffer()

    def _initialize_data(self) -> None:
        self._query_data = QueryData()
        self._document_datas = []

    def _add_to_buffer(
        self,
        query_data: QueryData,
        document_datas: List[DocumentData],
    ) -> None:
        self._query_data_buffer.append(query_data)
        self._document_data_buffer.extend(document_datas)

    def _flush_buffer(self) -> None:
        self._write_buffer_to_file_system(self._query_data_buffer)
        # TODO: implement write for document data
        self._initialize_data()

    def _write_buffer_to_file_system(self, buffer: Sequence[Data]) -> None:
        batch = pyarrow.RecordBatch.from_pylist([asdict(data) for data in buffer])
        schema = buffer[0].pyarrow_schema()
        with self._file_system.open_output_stream(self._data_path) as sink:
            writer = pyarrow.ipc.new_stream(sink, schema)
            writer.write_batch(batch)
            writer.close()
