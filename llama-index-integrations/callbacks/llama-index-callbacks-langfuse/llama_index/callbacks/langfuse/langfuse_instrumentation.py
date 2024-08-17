from inspect import BoundArguments
from threading import Lock
from types import GeneratorType
from typing import Optional, List, Any, Callable, Dict, Tuple, cast
from uuid import uuid4

import httpx
from langfuse.client import (
    StatefulTraceClient,
    StatefulGenerationClient,
    StatefulClient,
)
from llama_index.core.instrumentation.span import BaseSpan
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from llama_index.core.utilities.token_counting import TokenCounter
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.llms import LLM
from pydantic.v1 import Extra

from langfuse.utils.base_callback_handler import LangfuseBaseCallbackHandler


class LangfuseSpan(BaseSpan):
    """Langfuse Span."""

    client: StatefulClient


class LangfuseSpanHandler(BaseSpanHandler[LangfuseSpan], extra=Extra.allow):
    """Langfuse span handler."""

    def __init__(
        self,
        *,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        debug: bool = False,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        trace_name: Optional[str] = None,
        release: Optional[str] = None,
        version: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Any] = None,
        threads: Optional[int] = None,
        flush_at: Optional[int] = None,
        flush_interval: Optional[int] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None,
        tokenizer: Optional[Callable[[str], list]] = None,
        enabled: Optional[bool] = None,
        httpx_client: Optional[httpx.Client] = None,
        sdk_integration: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._handler = LangfuseBaseCallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            debug=debug,
            session_id=session_id,
            user_id=user_id,
            trace_name=trace_name,
            release=release,
            version=version,
            tags=tags,
            metadata=metadata,
            threads=threads,
            flush_at=flush_at,
            flush_interval=flush_interval,
            max_retries=max_retries,
            timeout=timeout,
            enabled=enabled,
            httpx_client=httpx_client,
            sdk_integration=sdk_integration or "llama-index_callback",
        )

        self._token_counter = TokenCounter(tokenizer)
        self._span_map: Dict[str, List[LangfuseSpan]] = {}
        self._to_update: List[LangfuseSpan] = []
        self._trace: Optional[StatefulTraceClient] = None
        self._lock = Lock()

    def new_span(
        self,
        id_: str,
        bound_args: BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[LangfuseSpan]:
        """New span."""
        with self._lock:
            # create trace span if no parent ID
            if not parent_span_id:
                # create trace
                return self._get_trace_span(
                    id_, bound_args.arguments, instance, **kwargs
                )

            if (
                parent_span_id not in self._span_map
                or len(self._span_map[parent_span_id]) < 1
            ):
                return None

            # create parent instance
            parent = self._span_map[parent_span_id][-1].client

            if self._is_generation(instance):
                # create generation span
                return self._get_generation_span(
                    id_, instance, bound_args.arguments, parent
                )
            else:
                # create normal span
                return self._get_span(
                    id_, parent, bound_args.arguments, instance, **kwargs
                )

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional[LangfuseSpan]:
        """Exit span."""
        with self._lock:
            if id_ not in self._span_map or len(self._span_map[id_]) < 1:
                return None

            span = self._span_map[id_].pop()
            output, metadata = self._parse_output_metadata(instance, result)

            # check for generators (for query engine streaming)
            is_generator_type = isinstance(result, GeneratorType)

            if not is_generator_type:
                # update query stream responses
                self._update_responses(self._to_update, output, metadata)
            else:
                # if query streaming response
                self._to_update.append(span)
                return

            # if this is a root level trace
            if not span.parent_id:
                trace = cast(StatefulTraceClient, span.client)

                if len(self._span_map[id_]) > 0:
                    # contains a root-level generation
                    gen_client = self._span_map[id_].pop()
                    generation = cast(StatefulGenerationClient, gen_client.client)
                    generation.end(output=output, metadata=metadata)
                trace.update(output=output, metadata=metadata)

                return

            # if this is a span/generation
            span.client.end(output=output, metadata=metadata)

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Optional[LangfuseSpan]:
        """Drop span due to early exiting."""
        with self._lock:
            if id_ not in self._span_map or len(self._span_map[id_]) < 1:
                return None

            span = self._span_map[id_].pop()
            output = str(err) if err else "An error occurred."

            # if this is a root level trace
            if not span.parent_id:
                trace = cast(StatefulTraceClient, span.client)

                if len(self._span_map[id_]) > 0:
                    # contains a root-level generation
                    gen_client = self._span_map[id_].pop()
                    generation = cast(StatefulGenerationClient, gen_client.client)
                    generation.end(status_message=output, level="ERROR")

                trace.update(output=output)

                return

            # if this is a span/generation
            span.client.end(status_message=output, level="ERROR")

    def flush(self) -> None:
        """Flushes langfuse."""
        self._handler.flush()

    def _get_span(
        self,
        id_: str,
        parent: StatefulClient,
        args: Dict[str, Any],
        instance: Optional[Any],
        **kwargs: Any,
    ) -> LangfuseSpan:
        """Gets trace span."""
        # Get trace metadata from contextvars or use default values
        name = self._handler.trace_name or instance.__class__.__name__
        version = self._handler.version
        release = self._handler.release
        session_id = self._handler.session_id
        user_id = self._handler.user_id
        metadata = self._handler.metadata or {}
        tags = self._handler.tags

        # metadata
        if kwargs:
            metadata["additional_kwargs"] = kwargs
        metadata = metadata or None

        # get parent span/trace
        span_client = parent.span(
            id=id_,
            name=name,
            version=version,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            release=release,
            input=args,
        )

        # add span object to map
        span_obj = LangfuseSpan(id=id_, parent_id=parent.id, client=span_client)
        if id_ not in self._span_map:
            self._span_map[id_] = []
        self._span_map[id_].append(span_obj)

        return span_obj

    def _get_trace_span(
        self,
        id_: str,
        args: Dict[str, Any],
        instance: Optional[Any],
        **kwargs: Any,
    ) -> LangfuseSpan:
        """Gets trace."""
        # Get trace metadata from contextvars or use default values
        name = self._handler.trace_name or instance.__class__.__name__
        version = self._handler.version
        release = self._handler.release
        session_id = self._handler.session_id
        user_id = self._handler.user_id
        metadata = self._handler.metadata or {}
        tags = self._handler.tags

        # metadata
        if kwargs:
            metadata["additional_kwargs"] = kwargs
        metadata = metadata or None

        # begin langfuse trace
        trace = self._handler.langfuse.trace(
            id=id_,
            name=name,
            version=version,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            release=release,
            input=args,
        )

        self._trace = trace

        # check if generation
        if self._is_generation(instance):
            self._get_generation_span(
                id_, instance, args, trace, update_parent=True, new_id=True
            )

        # add span to map
        trace_span = LangfuseSpan(id_=id_, client=trace)
        if id_ not in self._span_map:
            self._span_map[id_] = []
        self._span_map[id_].append(trace_span)

        return trace_span

    def _get_generation_span(
        self,
        id_: str,
        instance: Optional[Any],
        args: Dict[str, Any],
        parent: StatefulClient,
        update_parent: bool = False,
        new_id: bool = False,
    ) -> LangfuseSpan:
        """Gets generation span."""
        # create generation
        model_name = getattr(instance, "model_name", None) or getattr(
            instance, "model", None
        )

        if "texts" in args:
            # get "texts" as input; everything else is metadata
            input_dict = {"texts": args["texts"]}  # only contains texts
            metadata_dict = {
                key: val for key, val in args.items() if key != "texts"
            }  # other keys

            # calculate usage
            chunks = args["texts"]
            token_count = sum(
                self._token_counter.get_string_tokens(chunk) for chunk in chunks
            )
        else:
            # set input dict to args
            input_dict = args
            metadata_dict = None
            token_count = 0

        usage = {
            "input": 0,
            "output": 0,
            "total": token_count or None,
        }

        if new_id:
            gen_id = uuid4()
        else:
            gen_id = id_

        generation = parent.generation(
            id_=gen_id,
            name=instance.__class__.__name__,
            input=input_dict,
            metadata=metadata_dict,
            model=model_name,
            usage=usage,
        )

        if update_parent:
            # update parent with input/metadata
            parent.update(input=input_dict, metadata=metadata_dict)

        # add to map
        gen_span = LangfuseSpan(id=id_, client=generation, parent_id=parent.id)
        if id_ not in self._span_map:
            self._span_map[id_] = []
        self._span_map[id_].append(gen_span)

        return gen_span

    def _parse_output_metadata(
        self, instance: Optional[Any], output: Optional[Any]
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Parses output. Returns (output, metadata)."""
        if not output or isinstance(output, StreamingAgentChatResponse):
            # both str(StreamingAgentChatResponse) and StreamingAgentChatResponse returns an empty string
            # workaround is to look at output in the LLM generation on Langfuse
            # fix later?
            return None, None

        if isinstance(instance, BaseQueryEngine) and "response" in output.__dict__:
            # sort out metadata and output data
            output_res = {"response": output.response}
            metadata_dict = {
                key: val for key, val in output.__dict__.items() if key != "response"
            }  # other keys
            return output_res, metadata_dict

        return {"output": output}, None

    def _is_generation(self, instance: Optional[Any]) -> bool:
        """Determines if an instance is a generation or not."""
        return isinstance(instance, (BaseEmbedding, LLM))

    def _update_responses(
        self,
        response_list: List[LangfuseSpan],
        output: Optional[Any],
        metadata: Optional[Any],
    ) -> None:
        """Updates responses."""
        while len(response_list) > 0:
            to_update_span = response_list.pop()
            if isinstance(to_update_span.client, StatefulTraceClient):
                to_update_span.client.update(output=output, metadata=metadata)
            else:
                to_update_span.client.end(output=output, metadata=metadata)
