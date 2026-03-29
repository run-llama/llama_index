"""Base schema for callback managers."""

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

# timestamp for callback events
TIMESTAMP_FORMAT = "%m/%d/%Y, %H:%M:%S.%f"

# base trace_id for the tracemap in callback_manager
BASE_TRACE_EVENT = "root"


class CBEventType(str, Enum):
    """
    Callback manager event types.

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
        CONTEXT_COMPACTION: Fires when the agent compacts its context window
            (summarizes and drops older messages). Carries pre/post token counts
            and the summary text so observability tools can compare agent behavior
            across the compaction boundary.

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
    CONTEXT_COMPACTION = "context_compaction"


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
    # CONTEXT_COMPACTION payloads
    PRE_COMPACTION_TOKEN_COUNT = "pre_compaction_token_count"  # tokens before compaction
    POST_COMPACTION_TOKEN_COUNT = "post_compaction_token_count"  # tokens after compaction
    COMPACTION_SUMMARY = "compaction_summary"  # summary text produced by compaction
    DROPPED_MESSAGE_COUNT = "dropped_message_count"  # number of messages removed


# events that will never have children events
LEAF_EVENTS = (CBEventType.CHUNKING, CBEventType.LLM, CBEventType.EMBEDDING)


@dataclass
class CBEvent:
    """Generic class to store event information."""

    event_type: CBEventType
    payload: Optional[Dict[str, Any]] = None
    time: str = ""
    id_: str = ""

    def __post_init__(self) -> None:
        """Init time and id if needed."""
        if not self.time:
            self.time = datetime.now().strftime(TIMESTAMP_FORMAT)
        if not self.id_:
            self.id = str(uuid.uuid4())


@dataclass
class EventStats:
    """Time-based Statistics for events."""

    total_secs: float
    average_secs: float
    total_count: int
