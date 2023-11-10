"""Base reader class."""
from abc import ABC
from typing import Any, Dict, Iterable, List

from llama_index.bridge.langchain import Document as LCDocument
from llama_index.bridge.pydantic import Field
from llama_index.schema import BaseComponent, Document


class BaseReader(ABC):
    """Utilities for loading data from a directory."""

    def lazy_load_data(self, *args: Any, **load_kwargs: Any) -> Iterable[Document]:
        """Load data from the input directory lazily."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide lazy_load_data method currently"
        )

    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
        return list(self.lazy_load_data(*args, **load_kwargs))

    def load_langchain_documents(self, **load_kwargs: Any) -> List[LCDocument]:
        """Load data in LangChain document format."""
        docs = self.load_data(**load_kwargs)
        return [d.to_langchain_format() for d in docs]


class BasePydanticReader(BaseReader, BaseComponent):
    """Serialiable Data Loader with Pydatnic."""

    is_remote: bool = Field(
        default=False,
        description="Whether the data is loaded from a remote API or a local file.",
    )

    class Config:
        arbitrary_types_allowed = True


class ReaderConfig(BaseComponent):
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
        return "LoaderConfig"
