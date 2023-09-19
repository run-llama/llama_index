from abc import ABC, abstractmethod
from typing import List, Optional

from llama_index.bridge.pydantic import Field

from llama_index.callbacks import CallbackManager
from llama_index.indices.query.schema import QueryBundle
from llama_index.schema import NodeWithScore, BaseComponent


class BaseNodePostprocessor(BaseComponent, ABC):
    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )

    class Config:
        arbitrary_types_allowed = True

    # implement class_name so users don't have to worry about it when extending
    @classmethod
    def class_name(cls) -> str:
        return "BaseNodePostprocessor"

    @abstractmethod
    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        pass
