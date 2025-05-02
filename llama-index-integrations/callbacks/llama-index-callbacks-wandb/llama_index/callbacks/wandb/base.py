import os
import shutil
from collections import defaultdict
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import (
    TIMESTAMP_FORMAT,
    CBEvent,
    CBEventType,
    EventPayload,
)
from llama_index.core.callbacks.token_counting import get_llm_token_counts
from llama_index.core.utilities.token_counting import TokenCounter
from llama_index.core.utils import get_tokenizer

if TYPE_CHECKING:
    from llama_index.core.indices import (
        ComposableGraph,
        GPTEmptyIndex,
        GPTKeywordTableIndex,
        GPTRAKEKeywordTableIndex,
        GPTSimpleKeywordTableIndex,
        GPTSQLStructStoreIndex,
        GPTTreeIndex,
        GPTVectorStoreIndex,
        SummaryIndex,
    )
    from llama_index.core.storage.storage_context import StorageContext

    from wandb import Settings as WBSettings  # type: ignore[attr-defined]
    from wandb.sdk.data_types import trace_tree

    IndexType = Union[
        ComposableGraph,
        GPTKeywordTableIndex,
        GPTSimpleKeywordTableIndex,
        GPTRAKEKeywordTableIndex,
        SummaryIndex,
        GPTEmptyIndex,
        GPTTreeIndex,
        GPTVectorStoreIndex,
        GPTSQLStructStoreIndex,
    ]


# remove this class
class WandbRunArgs(TypedDict):
    job_type: Optional[str]
    dir: Optional[str]
    config: Union[Dict, str, None]
    project: Optional[str]
    entity: Optional[str]
    reinit: Optional[bool]
    tags: Optional[Sequence]
    group: Optional[str]
    name: Optional[str]
    notes: Optional[str]
    magic: Optional[Union[dict, str, bool]]
    config_exclude_keys: Optional[List[str]]
    config_include_keys: Optional[List[str]]
    anonymous: Optional[str]
    mode: Optional[str]
    allow_val_change: Optional[bool]
    resume: Optional[Union[bool, str]]
    force: Optional[bool]
    tensorboard: Optional[bool]
    sync_tensorboard: Optional[bool]
    monitor_gym: Optional[bool]
    save_code: Optional[bool]
    id: Optional[str]
    settings: Union["WBSettings", Dict[str, Any], None]


class WandbCallbackHandler(BaseCallbackHandler):
    """
    Callback handler that logs events to wandb.

    NOTE: this is a beta feature. The usage within our codebase, and the interface
    may change.

    Use the `WandbCallbackHandler` to log trace events to wandb. This handler is
    useful for debugging and visualizing the trace events. It captures the payload of
    the events and logs them to wandb. The handler also tracks the start and end of
    events. This is particularly useful for debugging your LLM calls.

    The `WandbCallbackHandler` can also be used to log the indices and graphs to wandb
    using the `persist_index` method. This will save the indexes as artifacts in wandb.
    The `load_storage_context` method can be used to load the indexes from wandb
    artifacts. This method will return a `StorageContext` object that can be used to
    build the index, using `load_index_from_storage`, `load_indices_from_storage` or
    `load_graph_from_storage` functions.


    Args:
        event_starts_to_ignore (Optional[List[CBEventType]]): list of event types to
            ignore when tracking event starts.
        event_ends_to_ignore (Optional[List[CBEventType]]): list of event types to
            ignore when tracking event ends.

    """

    def __init__(
        self,
        run_args: Optional[WandbRunArgs] = None,
        tokenizer: Optional[Callable[[str], List]] = None,
        event_starts_to_ignore: Optional[List[CBEventType]] = None,
        event_ends_to_ignore: Optional[List[CBEventType]] = None,
    ) -> None:
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

        from llama_index.core.indices import (
            ComposableGraph,
            GPTEmptyIndex,
            GPTKeywordTableIndex,
            GPTRAKEKeywordTableIndex,
            GPTSimpleKeywordTableIndex,
            GPTSQLStructStoreIndex,
            GPTTreeIndex,
            GPTVectorStoreIndex,
            SummaryIndex,
        )

        self._IndexType = (
            ComposableGraph,
            GPTKeywordTableIndex,
            GPTSimpleKeywordTableIndex,
            GPTRAKEKeywordTableIndex,
            SummaryIndex,
            GPTEmptyIndex,
            GPTTreeIndex,
            GPTVectorStoreIndex,
            GPTSQLStructStoreIndex,
        )

        self._run_args = run_args
        # Check if a W&B run is already initialized; if not, initialize one
        self._ensure_run(should_print_url=(self._wandb.run is None))  # type: ignore[attr-defined]

        self._event_pairs_by_id: Dict[str, List[CBEvent]] = defaultdict(list)
        self._cur_trace_id: Optional[str] = None
        self._trace_map: Dict[str, List[str]] = defaultdict(list)

        self.tokenizer = tokenizer or get_tokenizer()
        self._token_counter = TokenCounter(tokenizer=self.tokenizer)

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
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """
        Store event start data by event type.

        Args:
            event_type (CBEventType): event type to store.
            payload (Optional[Dict[str, Any]]): payload to store.
            event_id (str): event id to store.
            parent_id (str): parent event id.

        """
        event = CBEvent(event_type, payload=payload, id_=event_id)
        self._event_pairs_by_id[event.id_].append(event)
        return event.id_

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Store event end data by event type.

        Args:
            event_type (CBEventType): event type to store.
            payload (Optional[Dict[str, Any]]): payload to store.
            event_id (str): event id to store.

        """
        event = CBEvent(event_type, payload=payload, id_=event_id)
        self._event_pairs_by_id[event.id_].append(event)
        self._trace_map = defaultdict(list)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Launch a trace."""
        self._trace_map = defaultdict(list)
        self._cur_trace_id = trace_id
        self._start_time = datetime.now()

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        # Ensure W&B run is initialized
        self._ensure_run()

        self._trace_map = trace_map or defaultdict(list)
        self._end_time = datetime.now()

        # Log the trace map to wandb
        # We can control what trace ids we want to log here.
        self.log_trace_tree()

        # TODO (ayulockin): Log the LLM token counts to wandb when weave is ready

    def log_trace_tree(self) -> None:
        """Log the trace tree to wandb."""
        try:
            child_nodes = self._trace_map["root"]
            root_span = self._convert_event_pair_to_wb_span(
                self._event_pairs_by_id[child_nodes[0]],
                trace_id=self._cur_trace_id if len(child_nodes) > 1 else None,
            )

            if len(child_nodes) == 1:
                child_nodes = self._trace_map[child_nodes[0]]
                root_span = self._build_trace_tree(child_nodes, root_span)
            else:
                root_span = self._build_trace_tree(child_nodes, root_span)
            if root_span:
                root_trace = self._trace_tree.WBTraceTree(root_span)
                if self._wandb.run:  # type: ignore[attr-defined]
                    self._wandb.run.log({"trace": root_trace})  # type: ignore[attr-defined]
                self._wandb.termlog("Logged trace tree to W&B.")  # type: ignore[attr-defined]
        except Exception as e:
            print(f"Failed to log trace tree to W&B: {e}")
            # ignore errors to not break user code

    def persist_index(
        self, index: "IndexType", index_name: str, persist_dir: Union[str, None] = None
    ) -> None:
        """
        Upload an index to wandb as an artifact. You can learn more about W&B
        artifacts here: https://docs.wandb.ai/guides/artifacts.

        For the `ComposableGraph` index, the root id is stored as artifact metadata.

        Args:
            index (IndexType): index to upload.
            index_name (str): name of the index. This will be used as the artifact name.
            persist_dir (Union[str, None]): directory to persist the index. If None, a
                temporary directory will be created and used.

        """
        if persist_dir is None:
            persist_dir = f"{self._wandb.run.dir}/storage"  # type: ignore
            _default_persist_dir = True
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)

        if isinstance(index, self._IndexType):
            try:
                index.storage_context.persist(persist_dir)  # type: ignore

                metadata = None
                # For the `ComposableGraph` index, store the root id as metadata
                if isinstance(index, self._IndexType[0]):
                    root_id = index.root_id
                    metadata = {"root_id": root_id}

                self._upload_index_as_wb_artifact(persist_dir, index_name, metadata)
            except Exception as e:
                # Silently ignore errors to not break user code
                self._print_upload_index_fail_message(e)

        # clear the default storage dir
        if _default_persist_dir:
            shutil.rmtree(persist_dir, ignore_errors=True)

    def load_storage_context(
        self, artifact_url: str, index_download_dir: Union[str, None] = None
    ) -> "StorageContext":
        """
        Download an index from wandb and return a storage context.

        Use this storage context to load the index into memory using
        `load_index_from_storage`, `load_indices_from_storage` or
        `load_graph_from_storage` functions.

        Args:
            artifact_url (str): url of the artifact to download. The artifact url will
                be of the form: `entity/project/index_name:version` and can be found in
                the W&B UI.
            index_download_dir (Union[str, None]): directory to download the index to.

        """
        from llama_index.core.storage.storage_context import StorageContext

        artifact = self._wandb.use_artifact(artifact_url, type="storage_context")  # type: ignore[attr-defined]
        artifact_dir = artifact.download(root=index_download_dir)

        return StorageContext.from_defaults(persist_dir=artifact_dir)

    def _upload_index_as_wb_artifact(
        self, dir_path: str, artifact_name: str, metadata: Optional[Dict]
    ) -> None:
        """Utility function to upload a dir to W&B as an artifact."""
        artifact = self._wandb.Artifact(artifact_name, type="storage_context")  # type: ignore[attr-defined]

        if metadata:
            artifact.metadata = metadata

        artifact.add_dir(dir_path)
        self._wandb.run.log_artifact(artifact)  # type: ignore

    def _build_trace_tree(
        self, events: List[str], span: "trace_tree.Span"
    ) -> "trace_tree.Span":
        """Build the trace tree from the trace map."""
        for child_event in events:
            child_span = self._convert_event_pair_to_wb_span(
                self._event_pairs_by_id[child_event]
            )
            child_span = self._build_trace_tree(
                self._trace_map[child_event], child_span
            )
            span.add_child_span(child_span)

        return span

    def _convert_event_pair_to_wb_span(
        self,
        event_pair: List[CBEvent],
        trace_id: Optional[str] = None,
    ) -> "trace_tree.Span":
        """Convert a pair of events to a wandb trace tree span."""
        start_time_ms, end_time_ms = self._get_time_in_ms(event_pair)

        if trace_id is None:
            event_type = event_pair[0].event_type
            span_kind = self._map_event_type_to_span_kind(event_type)
        else:
            event_type = trace_id  # type: ignore
            span_kind = None

        wb_span = self._trace_tree.Span(
            name=f"{event_type}",
            span_kind=span_kind,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
        )

        inputs, outputs, wb_span = self._add_payload_to_span(wb_span, event_pair)
        wb_span.add_named_result(inputs=inputs, outputs=outputs)  # type: ignore

        return wb_span

    def _map_event_type_to_span_kind(
        self, event_type: CBEventType
    ) -> Union[None, "trace_tree.SpanKind"]:
        """Map a CBEventType to a wandb trace tree SpanKind."""
        if event_type == CBEventType.CHUNKING:
            span_kind = None
        elif event_type == CBEventType.NODE_PARSING:
            span_kind = None
        elif event_type == CBEventType.EMBEDDING:
            # TODO: add span kind for EMBEDDING when it's available
            span_kind = None
        elif event_type == CBEventType.LLM:
            span_kind = self._trace_tree.SpanKind.LLM
        elif event_type == CBEventType.QUERY:
            span_kind = self._trace_tree.SpanKind.AGENT
        elif event_type == CBEventType.AGENT_STEP:
            span_kind = self._trace_tree.SpanKind.AGENT
        elif event_type == CBEventType.RETRIEVE:
            span_kind = self._trace_tree.SpanKind.TOOL
        elif event_type == CBEventType.SYNTHESIZE:
            span_kind = self._trace_tree.SpanKind.CHAIN
        elif event_type == CBEventType.TREE:
            span_kind = self._trace_tree.SpanKind.CHAIN
        elif event_type == CBEventType.SUB_QUESTION:
            span_kind = self._trace_tree.SpanKind.CHAIN
        elif event_type == CBEventType.RERANKING:
            span_kind = self._trace_tree.SpanKind.CHAIN
        elif event_type == CBEventType.FUNCTION_CALL:
            span_kind = self._trace_tree.SpanKind.TOOL
        else:
            span_kind = None

        return span_kind

    def _add_payload_to_span(
        self, span: "trace_tree.Span", event_pair: List[CBEvent]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], "trace_tree.Span"]:
        """Add the event's payload to the span."""
        assert len(event_pair) == 2
        event_type = event_pair[0].event_type
        inputs = None
        outputs = None

        if event_type == CBEventType.NODE_PARSING:
            # TODO: disabled full detailed inputs/outputs due to UI lag
            inputs, outputs = self._handle_node_parsing_payload(event_pair)
        elif event_type == CBEventType.LLM:
            inputs, outputs, span = self._handle_llm_payload(event_pair, span)
        elif event_type == CBEventType.QUERY:
            inputs, outputs = self._handle_query_payload(event_pair)
        elif event_type == CBEventType.EMBEDDING:
            inputs, outputs = self._handle_embedding_payload(event_pair)

        return inputs, outputs, span

    def _handle_node_parsing_payload(
        self, event_pair: List[CBEvent]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Handle the payload of a NODE_PARSING event."""
        inputs = event_pair[0].payload
        outputs = event_pair[-1].payload

        if inputs and EventPayload.DOCUMENTS in inputs:
            documents = inputs.pop(EventPayload.DOCUMENTS)
            inputs["num_documents"] = len(documents)

        if outputs and EventPayload.NODES in outputs:
            nodes = outputs.pop(EventPayload.NODES)
            outputs["num_nodes"] = len(nodes)

        return inputs or {}, outputs or {}

    def _handle_llm_payload(
        self, event_pair: List[CBEvent], span: "trace_tree.Span"
    ) -> Tuple[Dict[str, Any], Dict[str, Any], "trace_tree.Span"]:
        """Handle the payload of a LLM event."""
        inputs = event_pair[0].payload
        outputs = event_pair[-1].payload

        assert isinstance(inputs, dict) and isinstance(outputs, dict)

        # Get `original_template` from Prompt
        if EventPayload.PROMPT in inputs:
            inputs[EventPayload.PROMPT] = inputs[EventPayload.PROMPT]

        # Format messages
        if EventPayload.MESSAGES in inputs:
            inputs[EventPayload.MESSAGES] = "\n".join(
                [str(x) for x in inputs[EventPayload.MESSAGES]]
            )

        token_counts = get_llm_token_counts(self._token_counter, outputs)
        metadata = {
            "formatted_prompt_tokens_count": token_counts.prompt_token_count,
            "prediction_tokens_count": token_counts.completion_token_count,
            "total_tokens_used": token_counts.total_token_count,
        }
        span.attributes = metadata

        # Make `response` part of `outputs`
        outputs = {EventPayload.RESPONSE: str(outputs[EventPayload.RESPONSE])}

        return inputs, outputs, span

    def _handle_query_payload(
        self, event_pair: List[CBEvent]
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Handle the payload of a QUERY event."""
        inputs = event_pair[0].payload
        outputs = event_pair[-1].payload

        if outputs:
            response_obj = outputs[EventPayload.RESPONSE]
            response = str(outputs[EventPayload.RESPONSE])

            if type(response).__name__ == "Response":
                response = response_obj.response
            elif type(response).__name__ == "StreamingResponse":
                response = response_obj.get_response().response
        else:
            response = " "

        outputs = {"response": response}

        return inputs, outputs

    def _handle_embedding_payload(
        self,
        event_pair: List[CBEvent],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        event_pair[0].payload
        outputs = event_pair[-1].payload

        chunks = []
        if outputs:
            chunks = outputs.get(EventPayload.CHUNKS, [])

        return {}, {"num_chunks": len(chunks)}

    def _get_time_in_ms(self, event_pair: List[CBEvent]) -> Tuple[int, int]:
        """Get the start and end time of an event pair in milliseconds."""
        start_time = datetime.strptime(event_pair[0].time, TIMESTAMP_FORMAT)
        end_time = datetime.strptime(event_pair[1].time, TIMESTAMP_FORMAT)

        start_time_in_ms = int(
            (start_time - datetime(1970, 1, 1)).total_seconds() * 1000
        )
        end_time_in_ms = int((end_time - datetime(1970, 1, 1)).total_seconds() * 1000)

        return start_time_in_ms, end_time_in_ms

    def _ensure_run(self, should_print_url: bool = False) -> None:
        """
        Ensures an active W&B run exists.

        If not, will start a new run with the provided run_args.
        """
        if self._wandb.run is None:  # type: ignore[attr-defined]
            # Make a shallow copy of the run args, so we don't modify the original
            run_args = self._run_args or {}  # type: ignore
            run_args: dict = {**run_args}  # type: ignore

            # Prefer to run in silent mode since W&B has a lot of output
            # which can be undesirable when dealing with text-based models.
            if "settings" not in run_args:  # type: ignore
                run_args["settings"] = {"silent": True}  # type: ignore

            # Start the run and add the stream table
            self._wandb.init(**run_args)  # type: ignore[attr-defined]
            self._wandb.run._label(repo="llama_index")  # type: ignore

            if should_print_url:
                self._print_wandb_init_message(
                    self._wandb.run.settings.run_url  # type: ignore
                )

    def _print_wandb_init_message(self, run_url: str) -> None:
        """Print a message to the terminal when W&B is initialized."""
        self._wandb.termlog(  # type: ignore[attr-defined]
            f"Streaming LlamaIndex events to W&B at {run_url}\n"
            "`WandbCallbackHandler` is currently in beta.\n"
            "Please report any issues to https://github.com/wandb/wandb/issues "
            "with the tag `llamaindex`."
        )

    def _print_upload_index_fail_message(self, e: Exception) -> None:
        """Print a message to the terminal when uploading the index fails."""
        self._wandb.termlog(  # type: ignore[attr-defined]
            f"Failed to upload index to W&B with the following error: {e}\n"
        )

    def finish(self) -> None:
        """Finish the callback handler."""
        self._wandb.finish()  # type: ignore[attr-defined]
