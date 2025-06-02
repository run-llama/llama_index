"""Simple reader that reads wikipedia."""

from typing import Any, List

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class WikipediaReader(BasePydanticReader):
    """
    Wikipedia reader.

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

    def load_data(
        self, pages: List[str], lang_prefix: str = "en", **load_kwargs: Any
    ) -> List[Document]:
        """
        Load data from the input directory.

        Args:
            pages (List[str]): List of pages to read.
            lang_prefix (str): Language prefix for Wikipedia. Defaults to English. Valid Wikipedia language codes
            can be found at https://en.wikipedia.org/wiki/List_of_Wikipedias.

        """
        import wikipedia

        if lang_prefix.lower() != "en":
            if lang_prefix.lower() in wikipedia.languages():
                wikipedia.set_lang(lang_prefix.lower())
            else:
                raise ValueError(
                    f"Language prefix '{lang_prefix}' for Wikipedia is not supported. Check supported languages at https://en.wikipedia.org/wiki/List_of_Wikipedias."
                )

        results = []
        for page in pages:
            wiki_page = wikipedia.page(page, **load_kwargs)
            page_content = wiki_page.content
            page_id = wiki_page.pageid
            results.append(Document(id_=page_id, text=page_content))
        return results
