from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from llama_index.utils import globals_helper
from llama_index.callbacks.base import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType


@dataclass
class TokenCountingEvent:
    prompt: str
    completion: str
    completion_token_count: int
    prompt_token_count: int
    total_token_count: int = 0
    event_id: str = ""

    def __post_init__(self) -> None:
        self.total_token_count = self.prompt_token_count + self.completion_token_count


class TokenCountingHandler(BaseCallbackHandler):
    """Callback handler for counting tokens in LLM and Embedding events.

    Args:
        tokenizer:
            Tokenizer to use. Defaults to the global tokenizer
            (see llama_index.utils.globals_helper).
        event_starts_to_ignore: List of event types to ignore at the start of a trace.
        event_ends_to_ignore: List of event types to ignore at the end of a trace.
    """

    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List]] = None,
        event_starts_to_ignore: Optional[List[CBEventType]] = None,
        event_ends_to_ignore: Optional[List[CBEventType]] = None,
    ) -> None:
        self.llm_token_counts: List[TokenCountingEvent] = []
        self.embedding_token_counts: List[TokenCountingEvent] = []
        self.tokenizer = tokenizer or globals_helper.tokenizer

        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore or [],
            event_ends_to_ignore=event_ends_to_ignore or [],
        )

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        return

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        return

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any
    ) -> str:
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any
    ) -> None:
        """Count the LLM or Embedding tokens as needed."""
        if (
            event_type == CBEventType.LLM
            and event_type not in self.event_ends_to_ignore
            and payload is not None
        ):
            self.llm_token_counts.append(
                TokenCountingEvent(
                    event_id=event_id,
                    prompt=payload.get("formatted_prompt", ""),
                    prompt_token_count=len(
                        self.tokenizer(payload.get("formatted_prompt", ""))
                    ),
                    completion=payload.get("response", ""),
                    completion_token_count=len(
                        self.tokenizer(payload.get("response", ""))
                    ),
                )
            )
        elif (
            event_type == CBEventType.EMBEDDING
            and event_type not in self.event_ends_to_ignore
            and payload is not None
        ):
            for chunk in payload.get("chunks", []):
                self.embedding_token_counts.append(
                    TokenCountingEvent(
                        event_id=event_id,
                        prompt=chunk,
                        prompt_token_count=len(self.tokenizer(chunk)),
                        completion="",
                        completion_token_count=0,
                    )
                )

    @property
    def total_llm_token_count(self) -> int:
        """Get the current total LLM token count."""
        return sum([x.total_token_count for x in self.llm_token_counts])

    @property
    def prompt_llm_token_count(self) -> int:
        """Get the current total LLM prompt token count."""
        return sum([x.prompt_token_count for x in self.llm_token_counts])

    @property
    def completion_llm_token_count(self) -> int:
        """Get the current total LLM completion token count."""
        return sum([x.completion_token_count for x in self.llm_token_counts])

    @property
    def total_embedding_token_count(self) -> int:
        """Get the current total Embedding token count."""
        return sum([x.total_token_count for x in self.embedding_token_counts])

    def reset_counts(self) -> None:
        """Reset the token counts."""
        self.llm_token_counts = []
        self.embedding_token_counts = []
