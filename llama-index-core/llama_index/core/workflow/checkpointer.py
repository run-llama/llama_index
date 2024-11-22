import uuid
from llama_index.core.bridge.pydantic import (
    BaseModel,
    Field,
    ConfigDict,
)
from typing import Optional, Dict, Any, List
from llama_index.core.workflow import (
    Workflow,
    Context,
    JsonSerializer,
)
from llama_index.core.workflow.context_serializers import BaseSerializer
from llama_index.core.workflow.handler import WorkflowHandler
from .events import Event


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
        workflow: Workflow,
        checkpoint_serializer: Optional[BaseSerializer] = None,
    ):
        self.workflow = workflow
        self.checkpoints = Dict[str, List[Checkpoint]] = {}
        self._checkpoint_serializer = checkpoint_serializer or JsonSerializer()

    def run(self) -> WorkflowHandler:
        ...

    def run_from(self, checkpoint: Checkpoint) -> WorkflowHandler:
        handler = self.workflow.run_from(checkpoint, self)

    def get_run_result(self, run_id: str) -> Any:
        ...

    def get_run(self, run_id: str) -> List[Checkpoint]:
        ...

    @property
    def checkpoints(self) -> Dict[str, List[Checkpoint]]:
        return self._checkpoints

    def _create_checkpoint(
        self,
        last_completed_step: Optional[str],
        input_ev: Optional[Event],
        output_ev: Event,
        ctx: Context,
    ) -> None:
        """Build a checkpoint around the last completed step."""
        try:
            run_id = output_ev.run_id
        except AttributeError:
            raise MissingWorkflowRunIdError(
                "No run_id found from `output_ev`. Make sure that step "
                f"with name '{last_completed_step}' returns an Event type."
            )

        if input_ev and run_id != input_ev.run_id:
            raise RunIdMismatchError(
                "run_id on input_ev and output_ev do not match, indicating "
                "that these event's don't belong to the same run."
            )

        if run_id is None:
            raise MissingWorkflowRunIdError(
                "No run_id found from input_ev nor output_ev."
            )

        checkpoint = Checkpoint(
            last_completed_step=last_completed_step,
            input_event=input_ev,
            output_event=output_ev,
            ctx_state=ctx.to_dict(serializer=self._checkpoint_serializer),
        )
        if run_id in self._checkpoints:
            self._checkpoints[run_id].append(checkpoint)
        else:
            self._checkpoints[run_id] = [checkpoint]

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


wflow_ckptr = WorkflowCheckpointer(workflow=w)
handler = wflow_ckptr.run_from(...)
await handler
