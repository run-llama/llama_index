"""Base reader class."""
from abc import abstractmethod
from typing import Any, Dict, List

try:
    from pydantic.v1 import Field
except ImportError:
    from pydantic import Field

from llama_index.bridge.langchain import Document as LCDocument
from llama_index.schema import Document, BaseComponent


class BaseReader:
    """Utilities for loading data from a directory."""

    @abstractmethod
    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""

    def load_langchain_documents(self, **load_kwargs: Any) -> List[LCDocument]:
        """Load data in LangChain document format."""
        docs = self.load_data(**load_kwargs)
        return [d.to_langchain_format() for d in docs]


class PydanticBaseReader(BaseReader, BaseComponent):
    """Serialiable Data Loader with Pydatnic."""

    is_remote: bool = False

    class Config:
        arbitrary_types_allowed = True


class LoaderConfig(BaseComponent):
    """Represents a loader and it's input arguments."""

    loader: BaseReader = Field(..., description="Loader to use.")
    loader_args: List[Any] = Field(default_factor=list, description="Loader args.")
    loader_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Loader kwargs."
    )

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def class_name(cls) -> str:
        """Get the name identifier of the class."""
        return "LoaderConfig"
