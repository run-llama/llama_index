"""Query-aware, token-budget web reader powered by dompruner."""

import asyncio
import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


class _Section(NamedTuple):
    """A heading-delimited markdown section with its heading provenance."""

    heading_path: List[str]
    section_title: Optional[str]
    text: str


def _import_dompruner() -> Tuple[Any, Any]:
    """Import dompruner lazily so the reader is importable without it."""
    try:
        from dompruner import run_pipeline, sync_run
    except ImportError as exc:
        raise ImportError(
            "DomPrunerWebReader requires the `dompruner` package. "
            "Install it with `pip install dompruner`."
        ) from exc
    return run_pipeline, sync_run


def _estimate_tokens(text: str) -> int:
    """Estimate token count at roughly 4 characters per token."""
    return max(1, len(text) // 4)


def _is_closing_fence(line: str, fence_char: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= 3 and set(stripped) == {fence_char}


def _split_sections(markdown: str) -> List[_Section]:
    """
    Split markdown into sections at heading boundaries.

    Fenced code blocks are tracked so a heading-looking line inside a code
    block never starts a new section and code blocks are never split apart.
    Every section records the ancestor heading chain (H1 -> H2 -> H3) that
    leads to it, keeping pruned chunks retrievable.
    """
    sections: List[_Section] = []
    heading_stack: List[Tuple[int, str]] = []
    current_lines: List[str] = []
    fence_char: Optional[str] = None

    def flush() -> None:
        text = "\n".join(current_lines).strip()
        current_lines.clear()
        if text:
            sections.append(
                _Section(
                    heading_path=[title for _, title in heading_stack],
                    section_title=heading_stack[-1][1] if heading_stack else None,
                    text=text,
                )
            )

    for line in markdown.splitlines():
        fence_match = _FENCE_RE.match(line)
        if fence_char is None:
            if fence_match:
                fence_char = fence_match.group(1)[0]
            else:
                heading_match = _HEADING_RE.match(line)
                if heading_match:
                    flush()
                    level = len(heading_match.group(1))
                    while heading_stack and heading_stack[-1][0] >= level:
                        heading_stack.pop()
                    heading_stack.append((level, heading_match.group(2).strip()))
        elif _is_closing_fence(line, fence_char):
            fence_char = None
        current_lines.append(line)

    flush()
    return sections


def _apply_token_budget(
    sections: List[_Section], token_budget: Optional[int]
) -> List[_Section]:
    """
    Keep the longest prefix of sections that fits the token budget.

    At least one section is always kept: sections are atomic (fenced code
    blocks are never split), so an oversized section is returned whole
    rather than truncated mid-signature.
    """
    if token_budget is None:
        return sections
    selected: List[_Section] = []
    used = 0
    for section in sections:
        cost = _estimate_tokens(section.text)
        if selected and used + cost > token_budget:
            break
        selected.append(section)
        used += cost
    return selected


def _build_documents(result: Any, sections: List[_Section]) -> List[Document]:
    base_metadata: Dict[str, Any] = dict(result.meta)
    base_metadata.update(
        {
            "url": result.url,
            "render_type": result.render_type,
            "original_tokens": result.original_tokens,
            "refined_tokens": result.refined_tokens,
            "reduction_ratio": result.reduction_ratio,
            "bm25_confidence": result.bm25_confidence,
        }
    )
    documents = []
    for section in sections:
        metadata = dict(base_metadata)
        metadata["section_title"] = section.section_title
        metadata["heading_path"] = section.heading_path
        documents.append(Document(text=section.text, metadata=metadata))
    return documents


class DomPrunerWebReader(BaseReader):
    """
    Query-aware, token-budget web page reader.

    Fetches pages through dompruner's zero-dependency pipeline (SSR DOM AST
    extraction, ``__NEXT_DATA__`` fast path, UA rotation on 403/429, and a
    Playwright fallback for true CSR pages) and returns one Document per
    heading-delimited section. Each Document carries its ancestor heading
    chain (H1 -> H2 -> H3) in ``metadata["heading_path"]``, so pruning a
    page never loses retrieval provenance.

    Processing happens in two separate stages:

    1. Extraction: deterministic content extraction that strips page
       furniture (nav, footer, cookie banners, template chrome).
    2. Pruning: when ``query`` is set, BM25 scores the sections of the
       clean content and the reader keeps only the relevant sections that
       fit ``token_budget``. Fenced code blocks are kept whole, so code
       signatures are never split.

    Args:
        query (Optional[str]): BM25 query for query-aware section
            selection. When None, the full extracted content is returned
            without pruning.
        token_budget (Optional[int]): Approximate token cap applied after
            BM25 selection when ``query`` is set. Defaults to 1500. Pass
            None to disable the cap. A single section larger than the
            budget is kept whole rather than split.

    """

    def __init__(
        self,
        query: Optional[str] = None,
        token_budget: Optional[int] = 1500,
    ) -> None:
        """Initialize with parameters."""
        self.query = query
        self.token_budget = token_budget

    def load_data(self, urls: List[str]) -> List[Document]:
        """
        Load data from the urls.

        Args:
            urls (List[str]): List of URLs to scrape.

        Returns:
            List[Document]: One document per retained section.

        """
        if not isinstance(urls, list):
            raise ValueError("urls must be a list of strings.")
        _, sync_run = _import_dompruner()
        return sync_run(self.async_load_data(urls))

    async def async_load_data(self, urls: List[str]) -> List[Document]:
        """
        Load data from the urls asynchronously.

        Args:
            urls (List[str]): List of URLs to scrape.

        Returns:
            List[Document]: One document per retained section.

        """
        if not isinstance(urls, list):
            raise ValueError("urls must be a list of strings.")
        run_pipeline, _ = _import_dompruner()
        query = self.query or ""
        results = await asyncio.gather(*(run_pipeline(url, query) for url in urls))

        documents = []
        for result in results:
            sections = _split_sections(result.markdown)
            if query:
                sections = _apply_token_budget(sections, self.token_budget)
            documents.extend(_build_documents(result, sections))
        return documents
