from typing import Any, List

from llama_index.core.chat_ui.models.artifact import Artifact
from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow.events import Event


class UIEvent(Event):
    type: str
    data: Any


class SourceNodesEvent(Event):
    nodes: List[NodeWithScore]


class ArtifactEvent(Event):
    type: str = "artifact"
    data: Artifact
