"""Node parser interface."""
from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Any, Optional

from llama_index.bridge.pydantic import BaseModel, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.schema import BaseComponent, BaseNode, Document


class NodeParser(BaseComponent, ABC):
    """Base interface for node parser."""

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Parse documents into nodes.

        Args:
            documents (Sequence[Document]): documents to parse

        """


class BaseExtractor(BaseComponent, ABC):
    """Base interface for feature extractor."""

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def extract(
        self,
        nodes: List[BaseNode],
    ) -> List[Dict]:
        """Post process nodes parsed from documents.

        Args:
            nodes (List[BaseNode]): nodes to extract from
        """


class NodeParserWithCallbackManager(NodeParser, ABC):
    """
    Base interface for node parser with callback manager.
    Workaround for CallbackManager not being a Pydantic object.
    """

    _callback_manager: CallbackManager = PrivateAttr()

    def __init__(self, callback_manager: Optional[CallbackManager] = None, **kwargs):
        super().__init__(**kwargs)
        self._callback_manager = callback_manager or CallbackManager()

    @property
    def callback_manager(self) -> CallbackManager:
        """Get callback manager."""
        return self._callback_manager

    @callback_manager.setter
    def callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        self._callback_manager = callback_manager

    def __setattr__(self, name: str, value: Any) -> None:
        whitelist = {"callback_manager"}
        if name in whitelist:
            super(BaseModel, self).__setattr__(name, value)
        else:
            super().__setattr__(name, value)
