"""Simple reader that reads wikipedia."""
from typing import Any, List

from llama_index.readers.base import BasePydanticReader
from llama_index.schema import Document


class WikipediaReader(BasePydanticReader):
    """Wikipedia reader.

    Reads a page.

    """

    is_remote: bool = True

    def __init__(self) -> None:
        """Initialize with parameters."""
        try:
            import wikipedia  # noqa
        except ImportError:
            raise ImportError(
                "`wikipedia` package not found, please run `pip install wikipedia`"
            )

    @classmethod
    def class_name(cls) -> str:
        return "WikipediaReader"

    def load_data(self, pages: List[str], **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory.

        Args:
            pages (List[str]): List of pages to read.

        """
        import wikipedia

        results = []
        for page in pages:
            page_content = wikipedia.page(page, **load_kwargs).content
            results.append(Document(text=page_content))
        return results
