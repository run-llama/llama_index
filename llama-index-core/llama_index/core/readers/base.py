"""Base reader class."""

from abc import ABC
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

if TYPE_CHECKING:
    from llama_index.core.bridge.langchain import Document as LCDocument
from llama_index.core.bridge.pydantic import Field
from llama_index.core.schema import BaseComponent, Document


class BaseReader(ABC):
    """Utilities for loading data from a directory."""

    def lazy_load_data(self, *args: Any, **load_kwargs: Any) -> Iterable[Document]:
        """Load data from the input directory lazily."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide lazy_load_data method currently"
        )

    async def alazy_load_data(
        self, *args: Any, **load_kwargs: Any
    ) -> Iterable[Document]:
        """Load data from the input directory lazily."""
        # Fake async - just calls the sync method. Override in subclasses for real async implementations.
        return self.lazy_load_data(*args, **load_kwargs)

    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
        return list(self.lazy_load_data(*args, **load_kwargs))

    async def aload_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
        return self.load_data(*args, **load_kwargs)

    def load_langchain_documents(self, **load_kwargs: Any) -> List["LCDocument"]:
        """Load data in LangChain document format."""
        docs = self.load_data(**load_kwargs)
        return [d.to_langchain_format() for d in docs]

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any], field: Optional[Any]):
        field_schema.update({"title": cls.__name__})

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema, handler
    ):  # Needed for pydantic v2 to work
        json_schema = handler(core_schema)
        json_schema = handler.resolve_ref_schema(json_schema)
        json_schema["title"] = cls.__name__
        return json_schema


class BasePydanticReader(BaseReader, BaseComponent):
    """Serialiable Data Loader with Pydantic."""

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
