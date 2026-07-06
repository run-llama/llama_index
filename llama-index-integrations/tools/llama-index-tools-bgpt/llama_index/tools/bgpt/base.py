"""BGPT tool spec."""

import json
from typing import Any, List, Optional

import requests
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

DEFAULT_SEARCH_URL = "https://bgpt.pro/api/mcp-search"
DEFAULT_LOOKUP_URL = "https://bgpt.pro/api/mcp-doi-lookup"

EVIDENCE_FIELDS = (
    "title",
    "doi",
    "publication_date",
    "publication_name",
    "one_sentence_summary",
    "methods_and_experimental_techniques",
    "sample_size_and_population",
    "results_and_conclusions",
    "paper_limitations_and_biases",
    "conflict_of_interest",
    "data_availability_statements",
    "code_and_data_links",
    "how_to_falsify",
    "study_blindspots",
)


def _paper_to_document(paper: dict[str, Any]) -> Document:
    """Convert a BGPT paper record into a LlamaIndex Document."""
    lines = []
    for field in EVIDENCE_FIELDS:
        value = paper.get(field)
        if value:
            lines.append(f"{field}: {value}")
    if not lines:
        lines.append(json.dumps(paper, indent=2))
    extra_info = {k: v for k, v in paper.items() if v is not None}
    return Document(
        text="\n".join(lines),
        extra_info=extra_info,
        metadata={"doi": paper.get("doi"), "title": paper.get("title")},
    )


class BGPTToolSpec(BaseToolSpec):
    """BGPT tool spec for structured scientific paper evidence retrieval."""

    spec_functions = ["search_papers", "lookup_paper"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        search_url: str = DEFAULT_SEARCH_URL,
        lookup_url: str = DEFAULT_LOOKUP_URL,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key
        self.search_url = search_url
        self.lookup_url = lookup_url
        self.timeout = timeout

    def search_papers(
        self,
        query: str,
        num_results: Optional[int] = 5,
        days_back: Optional[int] = None,
    ) -> List[Document]:
        """
        Search BGPT for scientific papers with structured full-text evidence.

        Args:
            query: Natural-language search query (e.g. "CAR-T response rates").
            num_results: Number of papers to return (default 5, max 100).
            days_back: Optional filter for papers published within the last N days.

        Returns:
            Documents with methods, sample sizes, limitations, conflicts of
            interest, falsifiability prompts, and other evidence fields.

        """
        payload: dict[str, Any] = {
            "query": query,
            "num_results": num_results or 5,
        }
        if days_back is not None:
            payload["days_back"] = days_back
        if self.api_key:
            payload["api_key"] = self.api_key

        response = requests.post(
            self.search_url,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        papers = data.get("results") or data.get("papers") or []
        return [_paper_to_document(paper) for paper in papers]

    def lookup_paper(self, doi: str) -> List[Document]:
        """
        Look up a single paper by DOI via BGPT.

        Args:
            doi: Paper DOI (e.g. "10.1038/s41586-024-07386-0").

        Returns:
            A single Document with structured evidence fields, or empty list if
            the paper is not found.

        """
        payload: dict[str, Any] = {"doi": doi}
        if self.api_key:
            payload["api_key"] = self.api_key

        response = requests.post(
            self.lookup_url,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        paper = data.get("result") or data.get("paper")
        if not paper:
            return []
        return [_paper_to_document(paper)]
