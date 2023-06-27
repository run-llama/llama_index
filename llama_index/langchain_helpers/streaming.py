from queue import Queue
from threading import Event
from typing import Any, Generator, Union

from llama_index.bridge.langchain import BaseCallbackHandler, LLMResult
from llama_index.callbacks.schema import CBEventType, EventPayload


class StreamingGeneratorCallbackHandler(BaseCallbackHandler):
    """Streaming callback handler."""

    def __init__(self, incomplete_payload, event_id, callback_manager) -> None:
        self._token_queue: Queue = Queue()
        self._done = Event()
        self._callback_manager = callback_manager
        self._prediction_token_count = 0
        self._generated_response = ""
        self._payload = incomplete_payload
        self._event_id = event_id

    def __deepcopy__(self, memo: Any) -> "StreamingGeneratorCallbackHandler":
        # NOTE: hack to bypass deepcopy in langchain
        return self

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        self._prediction_token_count += 1
        self._token_queue.put_nowait(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self._payload["prediction_tokens_count"] = self._prediction_token_count
        self._payload["total_tokens_used"] = (
            self._payload["formatted_prompt_tokens_count"]
            + self._prediction_token_count
        )
        self._payload[EventPayload.RESPONSE] = self._generated_response

        self._callback_manager.on_event_end(
            CBEventType.LLM,
            payload=self._payload,
            event_id=self._event_id,
        )
        self._done.set()

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        self._done.set()

    def get_response_gen(self) -> Generator:
        while True:
            if not self._token_queue.empty():
                token = self._token_queue.get_nowait()
                self._generated_response += token
                yield token
            elif self._done.is_set():
                break
