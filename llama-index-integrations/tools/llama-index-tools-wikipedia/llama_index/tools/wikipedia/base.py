"""Wikipedia tool spec."""

from typing import Any, Dict

from llama_index.core.tools.tool_spec.base import BaseToolSpec


class WikipediaToolSpec(BaseToolSpec):
    """
    Specifies two tools for querying information from Wikipedia.
    """

    spec_functions = ["load_data", "search_data"]

    def load_data(
        self, page: str, lang: str = "en", **load_kwargs: Dict[str, Any]
    ) -> str:
        """
        Retrieve a Wikipedia page. Useful for learning about a particular concept that isn't private information.

        Args:
            page (str): Title of the page to read.
            lang (str): Language of Wikipedia to read. (default: English)

        """
        import wikipedia

        wikipedia.set_lang(lang)
        try:
            wikipedia_page = wikipedia.page(page, **load_kwargs, auto_suggest=False)
        except wikipedia.PageError:
            return "Unable to load page. Try searching instead."
        return wikipedia_page.content

    def search_data(self, query: str, lang: str = "en") -> str:
        """
        Search Wikipedia for a page related to the given query.
        Use this tool when `load_data` returns no results.

        Args:
            query (str): the string to search for

        """
        import wikipedia

        pages = wikipedia.search(query)
        if len(pages) == 0:
            return "No search results."
        return self.load_data(pages[0], lang)
