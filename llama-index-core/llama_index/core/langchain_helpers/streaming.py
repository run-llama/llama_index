import time
from queue import Queue
from threading import Event
from typing import Any, Generator, List, Optional
from uuid import UUID

from llama_index.core.bridge.langchain import BaseCallbackHandler, LLMResult


class StreamingGeneratorCallbackHandler(BaseCallbackHandler):
    """Streaming callback handler."""

    def __init__(self) -> None:
        self._token_queue: Queue = Queue()
        self._done = Event()

    def __deepcopy__(self, memo: Any) -> "StreamingGeneratorCallbackHandler":
        # NOTE: hack to bypass deepcopy in langchain
        return self

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        self._token_queue.put_nowait(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self._done.set()

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self._done.set()

    def get_response_gen(self, timeout: float = 120.0) -> Generator:
        """Get response generator with timeout.

        Args:
            timeout (float): Maximum time in seconds to wait for the complete response.
                            Defaults to 120 seconds.
        """
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Response generation timed out after {timeout} seconds"
                )

            if not self._token_queue.empty():
                token = self._token_queue.get_nowait()
                yield token
            elif self._done.is_set():
                break
            else:
                # Small sleep to prevent CPU spinning
                time.sleep(0.01)
