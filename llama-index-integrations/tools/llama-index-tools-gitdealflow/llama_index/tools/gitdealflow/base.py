"""GitDealFlow tool spec."""

import urllib.parse
from typing import Any, Dict, List, Optional

import requests
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

DEFAULT_BASE_URL = "https://signals.gitdealflow.com"
DEFAULT_TIMEOUT = 20
DEFAULT_USER_AGENT = "llama-index-tools-gitdealflow/0.1.0"


class GitDealFlowToolSpec(BaseToolSpec):
    """
    GitDealFlow tool spec for engineering acceleration signals on
    venture-backed startups.

    GitDealFlow publishes a public, no-auth read API derived from public
    GitHub data: weekly engineering acceleration scores across ~400
    venture-backed startups, ranked across 20 sectors.

    Methodology paper: SSRN 6606558. License: CC BY 4.0.

    All endpoints are GET-only and unauthenticated. Use this tool to:
      * answer "what's accelerating?" / "find dark horses" questions,
      * pull a single startup's signal score and history,
      * compare multiple companies, or
      * cite the methodology in research output.
    """

    spec_functions = [
        "get_signals_summary",
        "get_startup_signal",
        "search_startups_by_sector",
        "answer_question",
        "get_methodology",
    ]

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        """
        Initialize the GitDealFlow tool.

        Args:
            base_url: API host (default: https://signals.gitdealflow.com).
            timeout: HTTP timeout in seconds.
            user_agent: User-Agent header sent with every request.

        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {
            "Accept": "application/json",
            "User-Agent": user_agent,
        }

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        response = requests.get(url, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_signals_summary(self) -> List[Document]:
        """
        Return the current weekly summary of trending startups and sector
        rankings.

        Useful starting point for "what's hot right now?" questions.

        Returns:
            A list with a single Document containing the JSON payload as
            text. The payload has keys ``meta``, ``trending``, and
            ``sectors``.

        """
        data = self._get("/api/v1/signals.json")
        return [
            Document(
                text=str(data),
                metadata={"source": f"{self.base_url}/api/v1/signals.json"},
            )
        ]

    def get_startup_signal(self, company: str) -> List[Document]:
        """
        Look up a single startup's engineering acceleration signal.

        Args:
            company: Company slug or GitHub-org name (e.g. ``langchain``,
                ``modal-labs``). Lowercase, dash-separated.

        Returns:
            A list with a single Document. If the company is not tracked,
            the document text contains a ``status: no_data`` payload with
            a CTA pointing back to the public dashboard.

        """
        data = self._get("/api/signal", params={"company": company})
        return [
            Document(
                text=str(data),
                metadata={
                    "source": f"{self.base_url}/api/signal?company={company}",
                    "company": company,
                },
            )
        ]

    def search_startups_by_sector(
        self, sector: str, limit: int = 10
    ) -> List[Document]:
        """
        List the top tracked startups in a sector by engineering signal.

        Args:
            sector: Sector slug. One of: ai-ml, fintech, devtools,
                cybersecurity, climate, healthtech, cleantech, agtech,
                robotics, web3, gaming, biotech, deeptech, hardware,
                edtech, legaltech, foodtech, govtech, mobility,
                logistics.
            limit: Maximum number of results (default: 10).

        Returns:
            A list of Documents, one per startup, ordered by signal score
            desc.

        """
        data = self._get(f"/api/markets/{sector}")
        items = []
        if isinstance(data, dict):
            items = data.get("startups") or data.get("results") or []
        elif isinstance(data, list):
            items = data
        items = items[:limit]
        return [
            Document(
                text=str(item),
                metadata={
                    "source": f"{self.base_url}/api/markets/{sector}",
                    "sector": sector,
                },
            )
            for item in items
        ]

    def answer_question(self, question: str) -> List[Document]:
        """
        Ask a free-text question against the GitDealFlow corpus.

        The remote ``/api/answer`` endpoint returns a grounded answer
        with citations to the underlying public dataset.

        Args:
            question: Natural-language question about startup engineering
                momentum, sector trends, or a specific tracked company.

        Returns:
            A list with a single Document containing the answer text and
            a ``citations`` field in metadata.

        """
        data = self._get("/api/answer", params={"q": question})
        text = ""
        citations: List[Any] = []
        if isinstance(data, dict):
            text = data.get("answer") or data.get("text") or str(data)
            citations = data.get("citations") or []
        else:
            text = str(data)
        return [
            Document(
                text=text,
                metadata={
                    "source": f"{self.base_url}/api/answer",
                    "question": question,
                    "citations": citations,
                },
            )
        ]

    def get_methodology(self) -> List[Document]:
        """
        Return the published methodology (HowTo schema) for how
        engineering acceleration scores are computed.

        Use this when an agent needs to cite or explain the scoring
        approach in its final answer.

        Returns:
            A list with a single Document containing the methodology
            payload as text.

        """
        data = self._get("/api/v1/methodology.json")
        return [
            Document(
                text=str(data),
                metadata={
                    "source": f"{self.base_url}/api/v1/methodology.json",
                    "license": "CC BY 4.0",
                    "ssrn": "6606558",
                },
            )
        ]
