import os
import shutil
from typing import TypedDict
from typing import Any, Dict, List, Optional, Sequence, Union, TYPE_CHECKING, Tuple
from collections import defaultdict
from datetime import datetime

from llama_index.callbacks.base import BaseCallbackHandler
from llama_index.callbacks.schema import (
    CBEvent,
    CBEventType,
    TIMESTAMP_FORMAT,
)

if TYPE_CHECKING:
    from wandb.sdk.data_types import trace_tree
    from wandb import Settings as WBSettings
    from llama_index.storage.storage_context import StorageContext

    from llama_index import (
        ComposableGraph,
        GPTKeywordTableIndex,
        GPTSimpleKeywordTableIndex,
        GPTRAKEKeywordTableIndex,
        GPTListIndex,
        GPTEmptyIndex,
        GPTTreeIndex,
        GPTVectorStoreIndex,
        GPTSQLStructStoreIndex,
    )

    IndexType = Union[
        ComposableGraph,
        GPTKeywordTableIndex,
        GPTSimpleKeywordTableIndex,
        GPTRAKEKeywordTableIndex,
        GPTListIndex,
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
        run_args: Optional[WandbRunArgs] = None,
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

        from llama_index import (
            ComposableGraph,
            GPTKeywordTableIndex,
            GPTSimpleKeywordTableIndex,
            GPTRAKEKeywordTableIndex,
            GPTListIndex,
            GPTEmptyIndex,
            GPTTreeIndex,
            GPTVectorStoreIndex,
            GPTSQLStructStoreIndex,
        )

        self._IndexType = (
            ComposableGraph,
            GPTKeywordTableIndex,
            GPTSimpleKeywordTableIndex,
            GPTRAKEKeywordTableIndex,
            GPTListIndex,
            GPTEmptyIndex,
            GPTTreeIndex,
            GPTVectorStoreIndex,
            GPTSQLStructStoreIndex,
        )

        self._run_args = run_args
        self._ensure_run(should_print_url=(self._wandb.run is None))

        self._event_pairs_by_id: Dict[str, List[CBEvent]] = defaultdict(list)
        self._cur_trace_id: Optional[str] = None
        self._trace_map: Dict[str, List[str]] = defaultdict(list)

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
        **kwargs: Any,
    ) -> str:
        """Store event start data by event type.

        Args:
            event_type (CBEventType): event type to store.
            payload (Optional[Dict[str, Any]]): payload to store.
            event_id (str): event id to store.

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
        """Store event end data by event type.

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

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        # Ensure W&B run is initialized
        self._ensure_run()

        # Shutdown the current trace
        self._trace_map = trace_map or defaultdict(list)

        # Log the trace map to wandb
        # We can control what trace ids we want to log here.
        self.log_trace_tree()

        # TODO (ayulockin): Log the LLM token counts to wandb when weave is ready

    def log_trace_tree(self) -> None:
        try:
            root_span = self._build_trace_tree()
            if root_span:
                root_trace = self._trace_tree.WBTraceTree(root_span)
                if self._wandb.run:
                    self._wandb.run.log({"trace": root_trace})
                self._wandb.termlog("Logged trace tree to W&B.")
        except:  # noqa
            # Silently ignore errors to not break user code
            pass

    def persist_index(
        self, index: "IndexType", index_name: str, persist_dir: Union[str, None] = None
    ) -> None:
        """Upload an index to wandb."""
        if persist_dir is None:
            persist_dir = f"{self._wandb.run.dir}/storage"  # type: ignore
            _default_persist_dir = True
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)

        if isinstance(index, self._IndexType):
            try:
                index.storage_context.persist(persist_dir)  # type: ignore
                self._upload_index_as_wb_artifact(persist_dir, index_name)
            except Exception as e:
                # Silently ignore errors to not break user code
                self._print_upload_index_fail_message(e)

        # clear the default storage dir
        if _default_persist_dir:
            shutil.rmtree(persist_dir, ignore_errors=True)

    def load_storage_context(
        self, artifact_url: str, index_download_dir: Union[str, None] = None
    ) -> "StorageContext":
        """Download an index from wandb and return a storage context."""
        from llama_index.storage.storage_context import StorageContext

        artifact = self._wandb.use_artifact(artifact_url, type="storage_context")
        artifact_dir = artifact.download(root=index_download_dir)

        storage_context = StorageContext.from_defaults(persist_dir=artifact_dir)
        return storage_context

    def _upload_index_as_wb_artifact(self, dir_path: str, artifact_name: str) -> None:
        artifact = self._wandb.Artifact(artifact_name, type="storage_context")
        artifact.add_dir(dir_path)
        self._wandb.run.log_artifact(artifact)  # type: ignore

    def _build_trace_tree(self) -> Union[None, "trace_tree.Span"]:
        root_span = None
        id_to_wb_span_tmp = {}
        for root_node, child_nodes in self._trace_map.items():
            if root_node == "root":
                # the payload for the first child node is the payload for the root_span
                root_span = self._convert_event_pair_to_wb_span(
                    self._event_pairs_by_id[child_nodes[0]],
                    trace_id=self._cur_trace_id if len(child_nodes) > 1 else None,
                )
                # add the child nodes to the root_span
                if len(child_nodes) > 1:
                    for child_node in child_nodes:
                        child_span = self._convert_event_pair_to_wb_span(
                            self._event_pairs_by_id[child_node]
                        )
                        root_span.add_child_span(child_span)
                        id_to_wb_span_tmp[child_node] = child_span
                else:
                    id_to_wb_span_tmp[child_nodes[0]] = root_span
            else:
                for child_node in child_nodes:
                    child_span = self._convert_event_pair_to_wb_span(
                        self._event_pairs_by_id[child_node]
                    )
                    id_to_wb_span_tmp[root_node].add_child_span(child_span)
                    id_to_wb_span_tmp[child_node] = child_span
                id_to_wb_span_tmp.pop(root_node)

        return root_span

    def _convert_event_pair_to_wb_span(
        self,
        event_pair: List[CBEvent],
        trace_id: Optional[str] = None,
    ) -> "trace_tree.Span":
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
        elif event_type == CBEventType.RETRIEVE:
            span_kind = self._trace_tree.SpanKind.TOOL
        elif event_type == CBEventType.SYNTHESIZE:
            span_kind = self._trace_tree.SpanKind.CHAIN
        elif event_type == CBEventType.TREE:
            span_kind = self._trace_tree.SpanKind.CHAIN
        else:
            raise ValueError(f"Unknown event type: {event_type}")

        return span_kind

    def _add_payload_to_span(
        self, span: "trace_tree.Span", event_pair: List[CBEvent]
    ) -> Tuple[
        Union[None, Dict[str, Any]], Union[None, Dict[str, Any]], "trace_tree.Span"
    ]:
        assert len(event_pair) == 2
        event_type = event_pair[0].event_type
        inputs = None
        outputs = None

        if event_type == CBEventType.NODE_PARSING:
            # parse input payload
            input_payload = event_pair[0].payload
            if input_payload:
                inputs = self._handle_node_parsing_payload(input_payload)
            # parse output payload
            output_payload = event_pair[-1].payload
            if output_payload:
                outputs = self._handle_node_parsing_payload(output_payload)
        elif event_type == CBEventType.LLM:
            inputs, outputs, span = self._handle_llm_payload(event_pair, span)
        else:
            inputs = event_pair[0].payload
            outputs = event_pair[-1].payload

        return inputs, outputs, span

    def _handle_node_parsing_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload_keys = payload.keys()
        assert "documents" or "nodes" in payload_keys

        if "documents" in payload_keys:
            stuffs = payload.get("documents", None)
            stuff_name = "documents"
        elif "nodes" in payload_keys:
            stuffs = payload.get("nodes", None)
            stuff_name = "nodes"

        if stuffs:
            tmp_str = ""
            for idx, stuff in enumerate(stuffs):
                tmp_str += f"**{stuff_name}**: {idx}\n" + stuff.text
                tmp_str += "\n*************** \n"
            return {"documents": tmp_str, "len_documents": len(stuffs)}
        else:
            return {}

    def _handle_llm_payload(
        self, event_pair: List[CBEvent], span: "trace_tree.Span"
    ) -> Tuple[Dict[str, Any], Dict[str, Any], "trace_tree.Span"]:
        inputs = event_pair[0].payload
        outputs = event_pair[-1].payload

        assert isinstance(inputs, dict) and isinstance(outputs, dict)

        # Make `formatted_prompt` part of `inputs`
        inputs["formatted_prompt"] = outputs.get("formatted_prompt", None)
        outputs.pop("formatted_prompt", None)

        # Make token counts part of span's `metadata`
        def filterByKey(keys: List[str]) -> Dict[str, int]:
            return {x: outputs[x] for x in keys}  # type: ignore

        metadata_keys = [
            "formatted_prompt_tokens_count",
            "prediction_tokens_count",
            "total_tokens_used",
        ]
        metadata = filterByKey(metadata_keys)
        span.attributes = metadata
        for meta_key in metadata_keys:
            outputs.pop(meta_key, None)

        # Make `response` part of `outputs`
        outputs = {"response": outputs["response"]}

        return inputs, outputs, span

    def _get_time_in_ms(self, event_pair: List[CBEvent]) -> Tuple[int, int]:
        start_time = datetime.strptime(event_pair[0].time, TIMESTAMP_FORMAT)
        end_time = datetime.strptime(event_pair[1].time, TIMESTAMP_FORMAT)

        start_time_in_ms = int(
            (start_time - datetime(1970, 1, 1)).total_seconds() * 1000
        )
        end_time_in_ms = int((end_time - datetime(1970, 1, 1)).total_seconds() * 1000)

        return start_time_in_ms, end_time_in_ms

    def _ensure_run(self, should_print_url: bool = False) -> None:
        """Ensures an active W&B run exists.

        If not, will start a new run with the provided run_args.
        """
        if self._wandb.run is None:
            # Make a shallow copy of the run args, so we don't modify the original
            run_args = self._run_args or {}  # type: ignore
            run_args: dict = {**run_args}  # type: ignore

            # Prefer to run in silent mode since W&B has a lot of output
            # which can be undesirable when dealing with text-based models.
            if "settings" not in run_args:  # type: ignore
                run_args["settings"] = {"silent": True}  # type: ignore

            # Start the run and add the stream table
            self._wandb.init(**run_args)

            if should_print_url:
                self._print_wandb_init_message(
                    self._wandb.run.settings.run_url  # type: ignore
                )

    def _print_wandb_init_message(self, run_url: str) -> None:
        self._wandb.termlog(
            f"Streaming LlamaIndex events to W&B at {run_url}\n"
            "`WandbCallbackHandler` is currently in beta.\n"
            "Please report any issues to https://github.com/wandb/wandb/issues "
            "with the tag `llamaindex`."
        )

    def _print_upload_index_fail_message(self, e: Exception) -> None:
        self._wandb.termlog(
            f"Failed to upload index to W&B with the following error: {e}\n"
        )

    def finish(self) -> None:
        """Finish the callback handler."""
        self._wandb.finish()
