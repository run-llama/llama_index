from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, cast

from llama_index.callbacks.base_handler import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.utilities.token_counting import TokenCounter
from llama_index.utils import get_tokenizer


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


def get_llm_token_counts(
    token_counter: TokenCounter, payload: Dict[str, Any], event_id: str = ""
) -> TokenCountingEvent:
    from llama_index.llms import ChatMessage

    if EventPayload.PROMPT in payload:
        prompt = str(payload.get(EventPayload.PROMPT))
        completion = str(payload.get(EventPayload.COMPLETION))

        return TokenCountingEvent(
            event_id=event_id,
            prompt=prompt,
            prompt_token_count=token_counter.get_string_tokens(prompt),
            completion=completion,
            completion_token_count=token_counter.get_string_tokens(completion),
        )

    elif EventPayload.MESSAGES in payload:
        messages = cast(List[ChatMessage], payload.get(EventPayload.MESSAGES, []))
        messages_str = "\n".join([str(x) for x in messages])

        response = payload.get(EventPayload.RESPONSE)
        response_str = str(response)

        # try getting attached token counts first
        try:
            usage = response.raw["usage"]  # type: ignore

            messages_tokens = usage.prompt_tokens
            response_tokens = usage.completion_tokens

            if messages_tokens == 0 or response_tokens == 0:
                raise ValueError("Invalid token counts!")

            return TokenCountingEvent(
                event_id=event_id,
                prompt=messages_str,
                prompt_token_count=messages_tokens,
                completion=response_str,
                completion_token_count=response_tokens,
            )

        except (ValueError, KeyError):
            # Invalid token counts, or no token counts attached
            pass

        # Should count tokens ourselves
        messages_tokens = token_counter.estimate_tokens_in_messages(messages)
        response_tokens = token_counter.get_string_tokens(response_str)

        return TokenCountingEvent(
            event_id=event_id,
            prompt=messages_str,
            prompt_token_count=messages_tokens,
            completion=response_str,
            completion_token_count=response_tokens,
        )
    else:
        raise ValueError(
            "Invalid payload! Need prompt and completion or messages and response."
        )


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
        verbose: bool = False,
    ) -> None:
        self.llm_token_counts: List[TokenCountingEvent] = []
        self.embedding_token_counts: List[TokenCountingEvent] = []
        self.tokenizer = tokenizer or get_tokenizer()

        self._token_counter = TokenCounter(tokenizer=self.tokenizer)
        self._verbose = verbose

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
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Count the LLM or Embedding tokens as needed."""
        if (
            event_type == CBEventType.LLM
            and event_type not in self.event_ends_to_ignore
            and payload is not None
        ):
            self.llm_token_counts.append(
                get_llm_token_counts(
                    token_counter=self._token_counter,
                    payload=payload,
                    event_id=event_id,
                )
            )

            if self._verbose:
                print(
                    "LLM Prompt Token Usage: "
                    f"{self.llm_token_counts[-1].prompt_token_count}\n"
                    "LLM Completion Token Usage: "
                    f"{self.llm_token_counts[-1].completion_token_count}",
                    flush=True,
                )
        elif (
            event_type == CBEventType.EMBEDDING
            and event_type not in self.event_ends_to_ignore
            and payload is not None
        ):
            total_chunk_tokens = 0
            for chunk in payload.get(EventPayload.CHUNKS, []):
                self.embedding_token_counts.append(
                    TokenCountingEvent(
                        event_id=event_id,
                        prompt=chunk,
                        prompt_token_count=self._token_counter.get_string_tokens(chunk),
                        completion="",
                        completion_token_count=0,
                    )
                )
                total_chunk_tokens += self.embedding_token_counts[-1].total_token_count

            if self._verbose:
                print(f"Embedding Token Usage: {total_chunk_tokens}", flush=True)

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
