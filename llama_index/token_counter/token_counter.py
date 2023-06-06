"""Token counter function."""

import asyncio
import logging
from contextlib import contextmanager
from typing import Any, Callable

from llama_index.indices.service_context import ServiceContext

logger = logging.getLogger(__name__)


def llm_token_counter(method_name_str: str) -> Callable:
    """
    Use this as a decorator for methods in index/query classes that make calls to LLMs.

    At the moment, this decorator can only be used on class instance methods with a
    `_llm_predictor` attribute.

    Do not use this on abstract methods.

    For example, consider the class below:
        .. code-block:: python
            class GPTTreeIndexBuilder:
            ...
            @llm_token_counter("build_from_text")
            def build_from_text(self, documents: Sequence[BaseDocument]) -> IndexGraph:
                ...

    If you run `build_from_text()`, it will print the output in the form below:

    ```
    [build_from_text] Total token usage: <some-number> tokens
    ```
    """

    def wrap(f: Callable) -> Callable:
        @contextmanager
        def wrapper_logic(_self: Any) -> Any:
            service_context = getattr(_self, "_service_context", None)
            if not isinstance(service_context, ServiceContext):
                raise ValueError(
                    "Cannot use llm_token_counter on an instance "
                    "without a service context."
                )
            llm_predictor = service_context.llm_predictor
            embed_model = service_context.embed_model

            start_token_ct = llm_predictor.total_tokens_used
            start_prediction_token_ct = llm_predictor.total_prediction_tokens_used
            start_prompt_token_ct = llm_predictor.total_prompt_tokens_used
            start_embed_token_ct = embed_model.total_tokens_used

            yield

            net_tokens = llm_predictor.total_tokens_used - start_token_ct
            llm_predictor.last_token_usage = net_tokens

            net_prediction_tokens = llm_predictor.prediction_token_usage - start_prediction_token_ct
            llm_predictor.last_prediction_token_usage = net_prediction_tokens

            net_prompt_tokens = llm_predictor.total_prompt_token_usage - start_prompt_token_ct
            llm_predictor.last_prompt_token_usage = net_prompt_tokens

            net_embed_tokens = embed_model.total_tokens_used - start_embed_token_ct
            embed_model.last_token_usage = net_embed_tokens

            # print outputs
            logger.info(
                f"> [{method_name_str}] Total LLM token usage: {net_tokens} tokens"
                f"> [{net_prompt_tokens} prompt tokens"
                f"> [{net_prediction_tokens} prediction tokens"
            )
            logger.info(
                f"> [{method_name_str}] Total embedding token usage: "
                f"{net_embed_tokens} tokens"
            )

        async def wrapped_async_llm_predict(
            _self: Any, *args: Any, **kwargs: Any
        ) -> Any:
            with wrapper_logic(_self):
                f_return_val = await f(_self, *args, **kwargs)

            return f_return_val

        def wrapped_llm_predict(_self: Any, *args: Any, **kwargs: Any) -> Any:
            with wrapper_logic(_self):
                f_return_val = f(_self, *args, **kwargs)

            return f_return_val

        if asyncio.iscoroutinefunction(f):
            return wrapped_async_llm_predict
        else:
            return wrapped_llm_predict

    return wrap
