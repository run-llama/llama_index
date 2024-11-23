import asyncio
import uuid
import functools
from llama_index.core.bridge.pydantic import (
    BaseModel,
    Field,
    ConfigDict,
)
from typing import (
    Optional,
    Dict,
    Any,
    List,
    Protocol,
    TYPE_CHECKING,
    Type,
    Awaitable,
)
from llama_index.core.workflow.context import Context
from llama_index.core.workflow.context_serializers import BaseSerializer, JsonSerializer
from llama_index.core.workflow.handler import WorkflowHandler
from .events import Event
from .errors import *

if TYPE_CHECKING:  # pragma: no cover
    from .workflow import Workflow


class CheckpointCallback(Protocol):
    checkpoints_config: Dict[str, bool]

    def __call__(
        self,
        last_completed_step: Optional[str],
        input_ev: Optional[Event],
        output_ev: Event,
        ctx: Context,
    ) -> Awaitable[None]:
        ...


class Checkpoint(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id_: str = Field(default_factory=lambda: str(uuid.uuid4()))
    last_completed_step: Optional[str]
    input_event: Optional[Event]
    output_event: Event
    ctx_state: Dict[str, Any]


class WorkflowCheckpointer:
    def __init__(
        self,
        workflow: "Workflow",
        checkpoint_serializer: Optional[BaseSerializer] = None,
        disabled_steps: List[str] = [],
    ):
        self._checkpoints: Dict[str, List[Checkpoint]] = {}
        self._checkpoint_serializer = checkpoint_serializer or JsonSerializer()
        self._lock: asyncio.Lock = asyncio.Lock()

        self.workflow = workflow
        self.checkpoints_config: Dict[str, bool] = {
            k: True for k in workflow._get_steps() if k != "_done"
        }
        for step_name in disabled_steps:
            self.disable_checkpoint(step_name)

    def enable_checkpoint(self, step: str) -> Dict[str, bool]:
        """Enable checkpointing after the completion of the specified step."""
        try:
            self.checkpoints_config[step] = True
        except KeyError:
            msg = f"This workflow does not contain a step with name {step}"
            raise WorkflowStepDoesNotExistError(msg)
        return {k: self.checkpoints_config[k] for k in [step]}

    def disable_checkpoint(self, step: str) -> Dict[str, bool]:
        """Disable checkpointing after the completion of the specified step."""
        try:
            self.checkpoints_config[step] = False
        except KeyError:
            msg = f"This workflow does not contain a step with name {step}"
            raise WorkflowStepDoesNotExistError(msg)
        return {k: self.checkpoints_config[k] for k in [step]}

    def generate_run_id(self) -> str:
        return str(uuid.uuid4())

    def run(self, **kwargs: Any) -> WorkflowHandler:
        return self.workflow.run(
            checkpoint_callback=self.new_checkpoint_callback_for_run(),
            **kwargs,
        )

    def run_from(self, checkpoint: Checkpoint, **kwargs: Any) -> WorkflowHandler:
        return self.workflow.run_from(
            checkpoint=checkpoint,
            ctx_serializer=self._checkpoint_serializer,
            checkpoint_callback=self.new_checkpoint_callback_for_run(),
            **kwargs,
        )

    @property
    def checkpoints(self) -> Dict[str, List[Checkpoint]]:
        return self._checkpoints

    def new_checkpoint_callback_for_run(self) -> CheckpointCallback:
        """Closure to generate a new `CheckpointCallback` with a unique run-id."""
        run_id = self.generate_run_id()

        async def _create_checkpoint(
            last_completed_step: Optional[str],
            input_ev: Optional[Event],
            output_ev: Event,
            ctx: Context,
        ) -> None:
            """Build a checkpoint around the last completed step."""
            checkpoint = Checkpoint(
                last_completed_step=last_completed_step,
                input_event=input_ev,
                output_event=output_ev,
                ctx_state=ctx.to_dict(serializer=self._checkpoint_serializer),
            )
            async with self._lock:
                if run_id in self.checkpoints:
                    self.checkpoints[run_id].append(checkpoint)
                else:
                    self.checkpoints[run_id] = [checkpoint]

        _create_checkpoint.checkpoints_config = self.checkpoints_config
        return _create_checkpoint

    def _checkpoint_filter_condition(
        self,
        ckpt: Checkpoint,
        last_completed_step: Optional[str],
        input_event_type: Optional[Type[Event]],
        output_event_type: Optional[Type[Event]],
    ) -> bool:
        if last_completed_step and ckpt.last_completed_step != last_completed_step:
            return False
        if input_event_type and type(ckpt.input_event) != input_event_type:
            return False
        if output_event_type and type(ckpt.output_event) != output_event_type:
            return False
        return True

    def filter_checkpoints(
        self,
        run_id: Optional[str] = None,
        last_completed_step: Optional[str] = None,
        input_event_type: Optional[Type[Event]] = None,
        output_event_type: Optional[Type[Event]] = None,
    ) -> List[Checkpoint]:
        """Returns a list of Checkpoint's based on user provided filters."""
        if (
            not run_id
            and not last_completed_step
            and not input_event_type
            and not output_event_type
        ):
            raise ValueError("Please specify a filter.")

        candidate_ckpts = (
            self.checkpoints[run_id]
            if run_id
            else functools.reduce(lambda a, b: a + b, self.checkpoints.values())
        )

        return [
            ckpt
            for ckpt in candidate_ckpts
            if self._checkpoint_filter_condition(
                ckpt, last_completed_step, input_event_type, output_event_type
            )
        ]
