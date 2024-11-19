import uuid
from llama_index.core.bridge.pydantic import BaseModel, Field
from .workflow import Workflow
from .handler import WorkflowHandler
from .context import Context
from typing import Any, List, Optional


class Checkpoint(BaseModel):
    id_: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prev_id: Optional[str] = Field(
        description="Id of previous checkpoint. None if there is no prev checkpoint."
    )
    original_run_id: str = Field(
        description="The id of the run this checkpoint originally emenated from of."
    )
    child_runs: List[str] = Field(
        default_factory=list,
        description="Run ids of the runs started off from this checkpoint.",
    )
    context_delta: Any = Field(
        description="The delta wrt to Context between this Checkpoint and its predecessor."
    )


class WorkflowProfiler:
    def __init__(self, workflow: Workflow):
        self.workflow = workflow
        self.checkpoints = List[Checkpoint] = []

    def _load_checkpoint(checkpoint: Checkpoint) -> Context:
        ...

    def run(self) -> WorkflowHandler:
        ...

    def run_from(self, checkpoint: Checkpoint) -> WorkflowHandler:
        ...

    def get_run_result(self, run_id: str) -> Any:
        ...

    def get_run(self, run_id: str) -> List[Checkpoint]:
        ...
