"""arXiv tool spec."""

from typing import Optional

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class ArxivToolSpec(BaseToolSpec):
    """arXiv tool spec."""

    spec_functions = ["arxiv_query"]

    def __init__(self, max_results: Optional[int] = 3):
        self.max_results = max_results

    def arxiv_query(self, query: str, sort_by: Optional[str] = "relevance"):
        """
        A tool to query arxiv.org
        ArXiv contains a variety of papers that are useful for answering
        mathematic and scientific questions.

        Args:
            query (str): The query to be passed to arXiv.
            sort_by (str): Either 'relevance' (default) or 'recent'

        """
        import arxiv

        sort = arxiv.SortCriterion.Relevance
        if sort_by == "recent":
            sort = arxiv.SortCriterion.SubmittedDate
        search = arxiv.Search(query, max_results=self.max_results, sort_by=sort)
        results = []
        for result in search.results():
            results.append(
                Document(text=f"{result.pdf_url}: {result.title}\n{result.summary}")
            )
        return results
