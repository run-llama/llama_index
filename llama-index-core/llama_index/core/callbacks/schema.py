"""Base schema for callback managers."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from deprecated import deprecated

from llama_index.core.bridge.pydantic import BaseModel

# timestamp for callback events
TIMESTAMP_FORMAT = "%m/%d/%Y, %H:%M:%S.%f"

# base trace_id for the tracemap in callback_manager
BASE_TRACE_EVENT = "root"


@deprecated("Since each event now has its own class now")
class CBEventType(str, Enum):
    """Callback manager event types.

    Attributes:
        CHUNKING: Logs for the before and after of text splitting.
        NODE_PARSING: Logs for the documents and the nodes that they are parsed into.
        EMBEDDING: Logs for the number of texts embedded.
        LLM: Logs for the template and response of LLM calls.
        QUERY: Keeps track of the start and end of each query.
        RETRIEVE: Logs for the nodes retrieved for a query.
        SYNTHESIZE: Logs for the result for synthesize calls.
        TREE: Logs for the summary and level of summaries generated.
        SUB_QUESTION: Logs for a generated sub question and answer.
    """

    CHUNKING = "chunking"
    NODE_PARSING = "node_parsing"
    EMBEDDING = "embedding"
    LLM = "llm"
    QUERY = "query"
    RETRIEVE = "retrieve"
    SYNTHESIZE = "synthesize"
    TREE = "tree"
    SUB_QUESTION = "sub_question"
    TEMPLATING = "templating"
    FUNCTION_CALL = "function_call"
    RERANKING = "reranking"
    EXCEPTION = "exception"
    AGENT_STEP = "agent_step"
    CUSTOM_EVENT = "custom_event"


class EventPayload(str, Enum):
    DOCUMENTS = "documents"  # list of documents before parsing
    CHUNKS = "chunks"  # list of text chunks
    NODES = "nodes"  # list of nodes
    PROMPT = "formatted_prompt"  # formatted prompt sent to LLM
    MESSAGES = "messages"  # list of messages sent to LLM
    COMPLETION = "completion"  # completion from LLM
    RESPONSE = "response"  # message response from LLM
    QUERY_STR = "query_str"  # query used for query engine
    SUB_QUESTION = "sub_question"  # a sub question & answer + sources
    EMBEDDINGS = "embeddings"  # list of embeddings
    TOP_K = "top_k"  # top k nodes retrieved
    ADDITIONAL_KWARGS = "additional_kwargs"  # additional kwargs for event call
    SERIALIZED = "serialized"  # serialized object for event caller
    FUNCTION_CALL = "function_call"  # function call for the LLM
    FUNCTION_OUTPUT = "function_call_response"  # function call output
    TOOL = "tool"  # tool used in LLM call
    MODEL_NAME = "model_name"  # model name used in an event
    TEMPLATE = "template"  # template used in LLM call
    TEMPLATE_VARS = "template_vars"  # template variables used in LLM call
    SYSTEM_PROMPT = "system_prompt"  # system prompt used in LLM call
    QUERY_WRAPPER_PROMPT = "query_wrapper_prompt"  # query wrapper prompt used in LLM
    EXCEPTION = "exception"  # exception raised in an event


# events that will never have children events
LEAF_EVENTS = (CBEventType.CHUNKING, CBEventType.LLM, CBEventType.EMBEDDING)


class CBEvent(BaseModel, ABC):
    """Base class to store event information."""

    _event_type: CBEventType
    _time: str = ""
    _id: str = ""

    def __init__(self, event_type: CBEventType) -> None:
        """Init time and id if needed."""
        self._event_type = event_type
        self._time = datetime.now().strftime(TIMESTAMP_FORMAT)
        self._id = str(uuid.uuid4())

    @property
    def id_(self) -> str:
        """Return the UUID id for the event."""
        return self._id

    @property
    def time(self) -> str:
        """Return the timestamp for the event."""
        return self._time

    @property
    @deprecated("You can call isinstance on the class to get the type of class/event.")
    def event_type(self) -> CBEventType:
        """Return the event type."""
        return self._event_type

    @property
    @abstractmethod
    @deprecated("You can access the payload properties directly from the class.")
    def payload(self) -> Any:
        """Return the payload for the event (to support legacy systems)."""


@dataclass
class EventStats:
    """Time-based Statistics for events."""

    total_secs: float
    average_secs: float
    total_count: int
