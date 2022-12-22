"""Simple reader that reads wikipedia."""
from typing import Any, List

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document


class WikipediaReader(BaseReader):
    """Wikipedia reader.

    Reads a page.

    """

    def __init__(self) -> None:
        """Initialize with parameters."""
        try:
            import wikipedia  # noqa: F401
        except ImportError:
            raise ValueError(
                "`wikipedia` package not found, please run `pip install wikipedia`"
            )

    def load_data(self, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory.

        Args:
            pages (List[str]): List of pages to read.

        """
        import wikipedia

        pages: List[str] = load_kwargs.pop("pages", None)
        if pages is None:
            raise ValueError('Must specify a "page" in `load_kwargs`.')

        results = []
        for page in pages:
            page_content = wikipedia.page(page).content
            results.append(Document(page_content))
        return results
