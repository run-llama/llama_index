"""Simple reader that ."""
from pathlib import Path
from typing import List, Any, Union

from gpt_index.readers.base import BaseReader
from gpt_index.schema import Document


class WikipediaReader(BaseReader):
    """Wikipedia reader.

    Reads a page.

    """

    def __init__(self) -> None:
        """Initialize with parameters."""
        try:
            import wikipedia
        except ImportError:
            raise ValueError(
                "`wikipedia` package not found, please run `pip install wikipedia`"
            )

    def load_data(self, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
        import wikipedia

        page = load_kwargs.pop("page", None)
        if page is None:
            raise ValueError("Must specify a \"page\" in `load_kwargs`.")

        page_content = wikipedia.page(page).content
        return [Document(page_content)]
