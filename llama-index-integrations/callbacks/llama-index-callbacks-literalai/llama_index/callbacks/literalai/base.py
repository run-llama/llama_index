import logging
from typing import Optional, cast

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.query import QueryEndEvent


def literalai_callback_handler(
    batch_size: int = 5,
    api_key: Optional[str] = None,
    url: Optional[str] = None,
    environment: Optional[str] = None,
    disabled: bool = False,
) -> BaseCallbackHandler:
    try:
        from literalai import LiteralClient
        from literalai.my_types import Environment

        literalai_client = LiteralClient(
            batch_size=batch_size,
            api_key=api_key,
            url=url,
            environment=cast(Environment, environment),
            disabled=disabled,
        )
        literalai_client.instrument_llamaindex()

        class QueryEndEventHandler(BaseEventHandler):
            """This handler will flush the Literal Client cache to Literal AI at the end of each query."""

            @classmethod
            def class_name(cls) -> str:
                """Class name."""
                return "QueryEndEventHandler"

            def handle(self, event: BaseEvent, **kwargs) -> None:
                """Flushes the Literal cache when receiving the QueryEnd event."""
                try:
                    if isinstance(event, QueryEndEvent):
                        literalai_client.flush()
                except Exception as e:
                    logging.error(
                        "Error in Literal AI global handler : %s",
                        str(e),
                        exc_info=True,
                    )

        dispatcher = get_dispatcher()
        event_handler = QueryEndEventHandler()
        dispatcher.add_event_handler(event_handler)

    except ImportError:
        raise ImportError(
            "Please install the Literal AI Python SDK with `pip install -U literalai`"
        )
