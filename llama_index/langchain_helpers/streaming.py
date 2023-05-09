import queue
from threading import Event
from typing import Any, Generator, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult


class StreamingGeneratorCallbackHandler(BaseCallbackHandler):
    """Streaming callback handler.
    """
    def __init__(self) -> None:
        self._token_queue = queue.Queue()
        self._done = Event()
    
    def __deepcopy__(self, memo: Any):
        # NOTE: hack to avoid skip deepcopy in langchain
        return self 

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        self._token_queue.put_nowait(token)
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self._done.set()

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        self._done.set()

    def get_response_gen(self) -> Generator:
        while True:
            if not self._token_queue.empty():
                token = self._token_queue.get(timeout=0.5)
                yield token
            elif self._done.is_set():
                break
            
            
            


