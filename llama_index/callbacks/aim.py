import logging
from typing import Any, Dict, List, Optional

try:
    from aim import Run, Text
except ModuleNotFoundError:
    Run, Text = None, None

from llama_index.callbacks.base_handler import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType, EventPayload

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class AimCallback(BaseCallbackHandler):
    """
    AimCallback callback class.

    Args:
        repo (:obj:`str`, optional):
            Aim repository path or Repo object to which Run object is bound.
            If skipped, default Repo is used.
        experiment_name (:obj:`str`, optional):
            Sets Run's `experiment` property. 'default' if not specified.
            Can be used later to query runs/sequences.
        system_tracking_interval (:obj:`int`, optional):
            Sets the tracking interval in seconds for system usage
            metrics (CPU, Memory, etc.). Set to `None` to disable
            system metrics tracking.
        log_system_params (:obj:`bool`, optional):
            Enable/Disable logging of system params such as installed packages,
            git info, environment variables, etc.
        capture_terminal_logs (:obj:`bool`, optional):
            Enable/Disable terminal stdout logging.
        event_starts_to_ignore (Optional[List[CBEventType]]):
            list of event types to ignore when tracking event starts.
        event_ends_to_ignore (Optional[List[CBEventType]]):
            list of event types to ignore when tracking event ends.
    """

    def __init__(
        self,
        repo: Optional[str] = None,
        experiment_name: Optional[str] = None,
        system_tracking_interval: Optional[int] = 1,
        log_system_params: Optional[bool] = True,
        capture_terminal_logs: Optional[bool] = True,
        event_starts_to_ignore: Optional[List[CBEventType]] = None,
        event_ends_to_ignore: Optional[List[CBEventType]] = None,
        run_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if Run is None:
            raise ModuleNotFoundError(
                "Please install aim to use the AimCallback: 'pip install aim'"
            )

        event_starts_to_ignore = (
            event_starts_to_ignore if event_starts_to_ignore else []
        )
        event_ends_to_ignore = event_ends_to_ignore if event_ends_to_ignore else []
        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore,
            event_ends_to_ignore=event_ends_to_ignore,
        )

        self.repo = repo
        self.experiment_name = experiment_name
        self.system_tracking_interval = system_tracking_interval
        self.log_system_params = log_system_params
        self.capture_terminal_logs = capture_terminal_logs
        self._run: Optional[Any] = None
        self._run_hash = None

        self._llm_response_step = 0

        self.setup(run_params)

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """
        Args:
            event_type (CBEventType): event type to store.
            payload (Optional[Dict[str, Any]]): payload to store.
            event_id (str): event id to store.
            parent_id (str): parent event id.
        """
        return ""

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            event_type (CBEventType): event type to store.
            payload (Optional[Dict[str, Any]]): payload to store.
            event_id (str): event id to store.
        """
        if not self._run:
            raise ValueError("AimCallback failed to init properly.")

        if event_type is CBEventType.LLM and payload:
            if EventPayload.PROMPT in payload:
                llm_input = str(payload[EventPayload.PROMPT])
                llm_output = str(payload[EventPayload.COMPLETION])
            else:
                message = payload.get(EventPayload.MESSAGES, [])
                llm_input = "\n".join([str(x) for x in message])
                llm_output = str(payload[EventPayload.RESPONSE])

            self._run.track(
                Text(llm_input),
                name="prompt",
                step=self._llm_response_step,
                context={"event_id": event_id},
            )

            self._run.track(
                Text(llm_output),
                name="response",
                step=self._llm_response_step,
                context={"event_id": event_id},
            )

            self._llm_response_step += 1
        elif event_type is CBEventType.CHUNKING and payload:
            for chunk_id, chunk in enumerate(payload[EventPayload.CHUNKS]):
                self._run.track(
                    Text(chunk),
                    name="chunk",
                    step=self._llm_response_step,
                    context={"chunk_id": chunk_id, "event_id": event_id},
                )

    @property
    def experiment(self) -> Run:
        if not self._run:
            self.setup()
        return self._run

    def setup(self, args: Optional[Dict[str, Any]] = None) -> None:
        if not self._run:
            if self._run_hash:
                self._run = Run(
                    self._run_hash,
                    repo=self.repo,
                    system_tracking_interval=self.system_tracking_interval,
                    log_system_params=self.log_system_params,
                    capture_terminal_logs=self.capture_terminal_logs,
                )
            else:
                self._run = Run(
                    repo=self.repo,
                    experiment=self.experiment_name,
                    system_tracking_interval=self.system_tracking_interval,
                    log_system_params=self.log_system_params,
                    capture_terminal_logs=self.capture_terminal_logs,
                )
                self._run_hash = self._run.hash

        # Log config parameters
        if args:
            try:
                for key in args:
                    self._run.set(key, args[key], strict=False)
            except Exception as e:
                logger.warning(f"Aim could not log config parameters -> {e}")

    def __del__(self) -> None:
        if self._run and self._run.active:
            self._run.close()

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        pass
