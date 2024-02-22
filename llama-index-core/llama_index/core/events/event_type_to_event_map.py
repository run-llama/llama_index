"""Dictionary to make CBEventType (which is technically a trace of a process) to a pair of start and end events."""

from typing import Dict, Literal

from llama_index.core.callbacks.schema import CBEvent, CBEventType
from llama_index.core.events.agent_step_events import (
    AgentStepEndEvent,
    AgentStepStartEvent,
)
from llama_index.core.events.chunking_events import ChunkingEndEvent, ChunkingStartEvent
from llama_index.core.events.embedding_events import (
    EmbeddingEndEvent,
    EmbeddingStartEvent,
)
from llama_index.core.events.exception_events import (
    ExceptionEndEvent,
    ExceptionStartEvent,
)
from llama_index.core.events.function_call_events import (
    FunctionCallEndEvent,
    FunctionCallStartEvent,
)
from llama_index.core.events.llm_events import LLMEndEvent, LLMStartEvent
from llama_index.core.events.node_parsing_events import (
    NodeParsingEndEvent,
    NodeParsingStartEvent,
)
from llama_index.core.events.query_events import QueryEndEvent, QueryStartEvent
from llama_index.core.events.reranking_events import (
    RerankingEndEvent,
    RerankingStartEvent,
)
from llama_index.core.events.retrieve_events import RetrieveEndEvent, RetrieveStartEvent
from llama_index.core.events.sub_question_events import (
    SubQuestionEndEvent,
    SubQuestionStartEvent,
)
from llama_index.core.events.templating_events import (
    TemplatingEndEvent,
    TemplatingStartEvent,
)
from llama_index.core.events.tree_events import TreeEndEvent, TreeStartEvent

event_type_to_event: Dict[CBEventType, Dict[Literal["start", "end"], type[CBEvent]]] = {
    CBEventType.CHUNKING: {"start": ChunkingStartEvent, "end": ChunkingEndEvent},
    CBEventType.NODE_PARSING: {
        "start": NodeParsingStartEvent,
        "end": NodeParsingEndEvent,
    },
    CBEventType.EMBEDDING: {
        "start": EmbeddingStartEvent,
        "end": EmbeddingEndEvent,
    },
    CBEventType.LLM: {
        "start": LLMStartEvent,
        "end": LLMEndEvent,
    },
    CBEventType.QUERY: {
        "start": QueryStartEvent,
        "end": QueryEndEvent,
    },
    CBEventType.RETRIEVE: {
        "start": RetrieveStartEvent,
        "end": RetrieveEndEvent,
    },
    CBEventType.SYNTHESIZE: {
        "start": RetrieveStartEvent,
        "end": RetrieveEndEvent,
    },
    CBEventType.TREE: {
        "start": TreeStartEvent,
        "end": TreeEndEvent,
    },
    CBEventType.SUB_QUESTION: {
        "start": SubQuestionStartEvent,
        "end": SubQuestionEndEvent,
    },
    CBEventType.TEMPLATING: {
        "start": TemplatingStartEvent,
        "end": TemplatingEndEvent,
    },
    CBEventType.FUNCTION_CALL: {
        "start": FunctionCallStartEvent,
        "end": FunctionCallEndEvent,
    },
    CBEventType.RERANKING: {
        "start": RerankingStartEvent,
        "end": RerankingEndEvent,
    },
    CBEventType.EXCEPTION: {
        "start": ExceptionStartEvent,
        "end": ExceptionEndEvent,
    },
    CBEventType.AGENT_STEP: {
        "start": AgentStepStartEvent,
        "end": AgentStepEndEvent,
    },
}
