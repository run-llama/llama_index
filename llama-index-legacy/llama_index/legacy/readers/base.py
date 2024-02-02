"""Base reader class."""

from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, Iterable, List

if TYPE_CHECKING:
    from llama_index.legacy.bridge.langchain import Document as LCDocument
from llama_index.legacy.bridge.pydantic import Field
from llama_index.legacy.schema import BaseComponent, Document


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

    def load_langchain_documents(self, **load_kwargs: Any) -> List["LCDocument"]:
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
    """Represents a reader and it's input arguments."""

    reader: BasePydanticReader = Field(..., description="Reader to use.")
    reader_args: List[Any] = Field(default_factory=list, description="Reader args.")
    reader_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Reader kwargs."
    )

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def class_name(cls) -> str:
        """Get the name identifier of the class."""
        return "ReaderConfig"

    def to_dict(self, **kwargs: Any) -> Dict[str, Any]:
        """Convert the class to a dictionary."""
        return {
            "loader": self.reader.to_dict(**kwargs),
            "reader_args": self.reader_args,
            "reader_kwargs": self.reader_kwargs,
            "class_name": self.class_name(),
        }

    def read(self) -> List[Document]:
        """Call the loader with the given arguments."""
        return self.reader.load_data(*self.reader_args, **self.reader_kwargs)
