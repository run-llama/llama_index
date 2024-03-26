import logging
from typing import List, Optional

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import (
    CBEventType,
)


class PythonicallyPrintingBaseHandler(BaseCallbackHandler):
    """
    Callback handler that prints logs in a Pythonic way. That is, not using `print` at all; use the logger instead.
    See https://stackoverflow.com/a/6918596/1147061 for why you should prefer using a logger over `print`.

    This class is meant to be subclassed, not used directly.

    Using this class, your LlamaIndex Callback Handlers can now make use of vanilla Python logging handlers now.
    One popular choice is https://rich.readthedocs.io/en/stable/logging.html#logging-handler.
    """

    def __init__(
        self,
        event_starts_to_ignore: Optional[List[CBEventType]] = None,
        event_ends_to_ignore: Optional[List[CBEventType]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger: Optional[logging.Logger] = logger
        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore,
            event_ends_to_ignore=event_ends_to_ignore,
        )

    def _print(self, str) -> None:
        if self.logger:
            self.logger.debug(str)
        else:
            # This branch is to preserve existing behavior.
            print(str, flush=True)
