import asyncio
import functools
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Type,
)

from llama_index.core.bridge.pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)
from llama_index.core.workflow.context_serializers import BaseSerializer, JsonSerializer
from llama_index.core.workflow.errors import WorkflowStepDoesNotExistError
from llama_index.core.workflow.events import Event

if TYPE_CHECKING:  # pragma: no cover
    from .context import Context
    from .handler import WorkflowHandler
    from .workflow import Workflow


class CheckpointCallback(Protocol):
    def __call__(
        self,
        run_id: str,
        last_completed_step: Optional[str],
        input_ev: Optional[Event],
        output_ev: Optional[Event],
        ctx: "Context",
    ) -> Awaitable[None]:
        ...


class Checkpoint(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id_: str = Field(default_factory=lambda: str(uuid.uuid4()))
    last_completed_step: Optional[str]
    input_event: Optional[Event]
    output_event: Optional[Event]
    ctx_state: Dict[str, Any]


class WorkflowCheckpointer:
    """An object that creates and maintain's checkpoints during a Workflow run.

    This checkpoint manager object works with multiple run's of a Workflow instance
    or from several different instances. Specified checkpoints can also be used
    as the starting point for a new Workflow run. Note that checkpoints are stored
    at the end of every step (with the exception of the _done step) for the attached
    Workflow.
    """

    def __init__(
        self,
        workflow: "Workflow",
        checkpoint_serializer: Optional[BaseSerializer] = None,
        disabled_steps: List[str] = [],
    ):
        """Create a WorkflowCheckpointer object.

        Args:
            workflow (Workflow): The wrapped workflow.
            checkpoint_serializer (Optional[BaseSerializer], optional): The serializer to use
                for serializing associated `Context` of a Workflow run. Defaults to None.
            disabled_steps (List[str], optional): Steps for which to disable checkpointing. Defaults to [].
        """
        self._checkpoints: Dict[str, List[Checkpoint]] = {}
        self._checkpoint_serializer = checkpoint_serializer or JsonSerializer()
        self._lock: asyncio.Lock = asyncio.Lock()

        self.workflow = workflow
        self.enabled_checkpoints: Set[str] = {
            k for k in workflow._get_steps() if k != "_done"
        }
        for step_name in disabled_steps:
            self.disable_checkpoint(step_name)

    def enable_checkpoint(self, step: str) -> None:
        """Enable checkpointing after the completion of the specified step."""
        if step not in self.workflow._get_steps():
            msg = f"This workflow does not contain a step with name {step}"
            raise WorkflowStepDoesNotExistError(msg)

        self.enabled_checkpoints.add(step)

    def disable_checkpoint(self, step: str) -> None:
        """Disable checkpointing after the completion of the specified step."""
        if step not in self.workflow._get_steps():
            msg = f"This workflow does not contain a step with name {step}"
            raise WorkflowStepDoesNotExistError(msg)
        try:
            self.enabled_checkpoints.remove(step)
        except KeyError:
            pass

    def run(self, **kwargs: Any) -> "WorkflowHandler":
        """Run the workflow with checkpointing."""
        return self.workflow.run(
            checkpoint_callback=self.new_checkpoint_callback_for_run(),
            **kwargs,
        )

    def run_from(self, checkpoint: Checkpoint, **kwargs: Any) -> "WorkflowHandler":
        """Run the attached workflow from the specified checkpoint."""
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

        async def _create_checkpoint(
            run_id: str,
            last_completed_step: Optional[str],
            input_ev: Optional[Event],
            output_ev: Optional[Event],
            ctx: "Context",
        ) -> None:
            """Build a checkpoint around the last completed step."""
            if last_completed_step not in self.enabled_checkpoints:
                return

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
