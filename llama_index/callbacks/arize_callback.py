"""
Callback handlers for logging to Arize and Phoenix-compatible file formats.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypeAlias

from llama_index.callbacks.base import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType, EventPayload

Embedding: TypeAlias = List[float]


@dataclass
class RetrievalAugmentedGenerationQueryData:
    query_text: Optional[str] = None
    query_embedding: Optional[Embedding] = None
    response_text: Optional[str] = None
    document_ids: List[str] = field(default_factory=list)
    document_hashes: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)


@dataclass
class RetrievalAugmentedGenerationDocumentData:
    document_text: Optional[str] = None
    document_embedding: Optional[Embedding] = None
    document_id: Optional[str] = None
    document_hash: Optional[str] = None


class BaseArizeCallbackHandler(BaseCallbackHandler, ABC):
    def __init__(self) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self._query_data = RetrievalAugmentedGenerationQueryData()
        self._document_datas: List[RetrievalAugmentedGenerationDocumentData] = []

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
                        RetrievalAugmentedGenerationDocumentData(
                            document_text=node.text,
                            document_id=node.id_,
                            document_hash=node.hash,
                        )
                    )
            elif event_type is CBEventType.LLM:
                self._query_data.response_text = payload[EventPayload.RESPONSE]

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""
        if trace_id == "query":
            self._query_data = RetrievalAugmentedGenerationQueryData()

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        if trace_id == "query":
            self._log_data()

    @abstractmethod
    def _log_data(self) -> None:
        ...

    def _initialize_data(self) -> None:
        self._query_data = RetrievalAugmentedGenerationQueryData()
        self._document_datas = []


class ArizeCallbackHandler(BaseArizeCallbackHandler):
    def _log_data(self) -> None:
        print(self._query_data)
        print(self._document_datas)


class OpenInferenceCallbackHandler(BaseArizeCallbackHandler):
    def _log_data(self) -> None:
        print(self._query_data)
        print(self._document_datas)
