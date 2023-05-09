import queue
from typing import Any, Dict, Generator, List, Optional, Union
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult


class StreamingGeneratorCallbackHandler(BaseCallbackHandler):
    """Streaming callback handler.
    """
    def __init__(self) -> None:
        self._token_queue = queue.Queue()
        self._done = False

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        self._token_queue = queue.Queue()
        self._done = False

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        self._token_queue.put_nowait(token)
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.done = True

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        self.done = True

    def get_response_gen(self) -> Generator:
        def gen():
            while not self.done or self._token_queue.not_empty:
                token = self._token_queue.get()
                yield token

        return gen()


