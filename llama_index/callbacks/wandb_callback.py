from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional

from llama_index.callbacks.base import BaseCallbackHandler
from llama_index.callbacks.schema import (
    CBEvent,
    CBEventType,
    TIMESTAMP_FORMAT,
)


class WandbCallbackHandler(BaseCallbackHandler):
    """Callback handler that logs events to wandb.

    NOTE: this is a beta feature. The usage within our codebase, and the interface
    may change.

    This handler simply keeps track of event starts/ends, separated by event types.
    You can use this callback handler to keep track of and debug events.

    Args:
        event_starts_to_ignore (Optional[List[CBEventType]]): list of event types to
            ignore when tracking event starts.
        event_ends_to_ignore (Optional[List[CBEventType]]): list of event types to
            ignore when tracking event ends.
    """

    def __init__(
        self,
        event_starts_to_ignore: Optional[List[CBEventType]] = None,
        event_ends_to_ignore: Optional[List[CBEventType]] = None,
    ) -> None:
        """Initialize the wandb handler."""
        try:
            import wandb
            from wandb.sdk.data_types import trace_tree

            self._wandb = wandb
            self._trace_tree = trace_tree
        except ImportError:
            raise ImportError(
                "WandbCallbackHandler requires wandb. "
                "Please install it with `pip install wandb`."
            )

        if self._wandb.run is None:
            # TODO: pass wandb args
            run = self._wandb.init(project="llama_index", settings={"silent": True})
            print("W&b run URL: ", run.url)

        self._events_by_id: Dict[str, List[CBEvent]] = {}
        self._llm_events = []

        self._cache_query_events = []
        self._is_query = False

        event_starts_to_ignore = (
            event_starts_to_ignore if event_starts_to_ignore else []
        )
        event_ends_to_ignore = event_ends_to_ignore if event_ends_to_ignore else []
        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore,
            event_ends_to_ignore=event_ends_to_ignore,
        )

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any
    ) -> str:
        """Store event start data by event type.

        Args:
            event_type (CBEventType): event type to store.
            payload (Optional[Dict[str, Any]]): payload to store.
            event_id (str): event id to store.

        """
        event = CBEvent(event_type, payload=payload, id_=event_id)
        self._events_by_id[event.id_] = [event]

        # Cache Query events
        if event_type == CBEventType.QUERY:
            self._is_query = True

        if self._is_query:
            self._cache_query_events.append(event)

        return event.id_

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any
    ) -> None:
        """Store event end data by event type.

        Args:
            event_type (CBEventType): event type to store.
            payload (Optional[Dict[str, Any]]): payload to store.
            event_id (str): event id to store.

        """
        event = CBEvent(event_type, payload=payload, id_=event_id)
        self._events_by_id[event.id_].append(event)

        assert len(self._events_by_id[event.id_]) == 2
        assert self._events_by_id[event.id_][0].event_type == self._events_by_id[event.id_][1].event_type

        if self._is_query:
            self._cache_query_events.append(event)

        if event_type == CBEventType.QUERY:
            query_event_pairs = self._get_event_pairs(self._cache_query_events)
            query_span = self._query_trace_tree(query_event_pairs)
            query_trace = self._trace_tree.WBTraceTree(query_span)

            if self._wandb.run is not None:
                self._wandb.run.log({"query_trace": query_trace})
    
            self._cache_query_events = []
            self._is_query = False

        # Log independent LLM events to trace view
        if event_type == CBEventType.LLM:
            self._llm_events.append(event)
            llm_usage_info = self._get_llm_usage_info(event.id_)

            llm_table = self._wandb.Table(
                columns=[
                    "event_id", "event_start_time", "event_end_time", "inputs", "outputs", "formatted_prompt_tokens_count", "prediction_tokens_count", "total_token_count"]
            )

            len(llm_usage_info)
            llm_table.add_data(*llm_usage_info)
            self._wandb.run.log({"llm_tracker": llm_table})

    def _get_llm_usage_info(self, llm_id):
        event_pair = self._events_by_id[llm_id]
        start_time, end_time = event_pair[0].time, event_pair[1].time

        # ["event_id", "event_start_time", "event_end_time", "inputs", "outputs", "total_token_count"]
        inputs = event_pair[0].payload
        inputs = [f"{k}\n******\n{v}" for k, v in inputs.items()]
        inputs = "\n".join(inputs)

        outputs = event_pair[1].payload
        outputs = [f"{k}\n******\n{v}" for k, v in outputs.items()]
        outputs = "\n".join(outputs)

        return [
            llm_id,
            start_time,
            end_time,
            inputs,
            outputs,
            event_pair[1].payload["formatted_prompt_tokens_count"],
            event_pair[1].payload["prediction_tokens_count"],
            event_pair[1].payload["total_tokens_used"],
        ]

    def _query_trace_tree(self, query_event_pairs):
        query_span = None
        retrive_span = None
        synth_span = None

        for event_pair in query_event_pairs:
            assert len(event_pair) == 2
            span = self._convert_event_to_wb_span(event_pair)
            span.add_named_result(
                inputs=event_pair[0].payload,
                outputs=event_pair[1].payload,
            )

            if event_pair[0].event_type == CBEventType.QUERY:
                query_span = span
            elif event_pair[0].event_type == CBEventType.RETRIEVE:
                retrive_span = span
                query_span.add_child_span(retrive_span)
            elif event_pair[0].event_type == CBEventType.SYNTHESIZE:
                synth_span = span
                query_span.add_child_span(synth_span)
            elif event_pair[0].event_type == CBEventType.EMBEDDING:
                retrive_span.add_child_span(span)
            elif event_pair[0].event_type == CBEventType.LLM:
                synth_span.add_child_span(span)

        return query_span

    def _convert_event_to_wb_span(self, event_pair):
        start_time_ms, end_time_ms = self._get_time_in_ms(event_pair)

        event_type = event_pair[0].event_type

        if event_type == CBEventType.QUERY:
            span_kind = self._trace_tree.SpanKind.CHAIN # read QUERY
        elif event_type == CBEventType.RETRIEVE:
            span_kind = self._trace_tree.SpanKind.AGENT # read RETRIEVE
        elif event_type == CBEventType.EMBEDDING:
            span_kind = self._trace_tree.SpanKind.TOOL # read EMBEDDING
        elif event_type == CBEventType.SYNTHESIZE:
            span_kind = self._trace_tree.SpanKind.AGENT # read SYNTHESIZE
        elif event_type == CBEventType.LLM:
            span_kind = self._trace_tree.SpanKind.LLM

        wb_span = self._trace_tree.Span(
            name=f"{event_type}",
            span_kind=span_kind,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
        )

        return wb_span

    def _get_time_in_ms(self, event_pair):
        start_time = datetime.strptime(event_pair[0].time, TIMESTAMP_FORMAT)
        end_time = datetime.strptime(event_pair[1].time, TIMESTAMP_FORMAT)

        start_time = int((start_time - datetime(1970,1,1)).total_seconds()*1000)
        end_time = int((end_time - datetime(1970,1,1)).total_seconds()*1000)

        return start_time, end_time

    def _get_event_pairs(self, events: List[CBEvent]) -> List[List[CBEvent]]:
        """Helper function to pair events according to their ID."""
        event_pairs: Dict[str, List[CBEvent]] = defaultdict(list)
        for event in events:
            event_pairs[event.id_].append(event)

        sorted_events = sorted(
            event_pairs.values(),
            key=lambda x: datetime.strptime(x[0].time, TIMESTAMP_FORMAT),
        )
        return sorted_events

    def finish(self) -> None:
        """Finish the callback handler."""
        self._wandb.finish()
