"""Reader for .causal knowledge graph files."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class CausalReader(BasePydanticReader):
    """
    Reader for .causal binary knowledge graph files.

    The .causal format provides deterministic inference with zero hallucination.
    It pre-computes transitive chains at storage time, enabling 30-40x faster
    queries than SQLite while amplifying facts by 50-200%.

    Args:
        include_inferred: Whether to include inferred triplets (default: True)
        min_confidence: Minimum confidence threshold (default: 0.0)

    Example:
        >>> from llama_index.readers.causal import CausalReader
        >>> reader = CausalReader()
        >>> documents = reader.load_data("knowledge.causal")
        >>> # Each document contains a triplet with metadata

    References:
        - PyPI: https://pypi.org/project/dotcausal/
        - GitHub: https://github.com/DT-Foss/dotcausal
        - Paper: https://doi.org/10.5281/zenodo.18326222
    """

    is_remote: bool = False
    include_inferred: bool = True
    min_confidence: float = 0.0

    def __init__(
        self,
        include_inferred: bool = True,
        min_confidence: float = 0.0,
    ) -> None:
        """Initialize CausalReader.

        Args:
            include_inferred: Include inferred (derived) triplets
            min_confidence: Minimum confidence threshold for triplets
        """
        try:
            import dotcausal  # noqa
        except ImportError:
            raise ImportError(
                "`dotcausal` package not found, please run `pip install dotcausal`"
            )

        super().__init__(
            include_inferred=include_inferred,
            min_confidence=min_confidence,
        )

    @classmethod
    def class_name(cls) -> str:
        return "CausalReader"

    def load_data(
        self,
        file_path: str,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        **load_kwargs: Any,
    ) -> List[Document]:
        """
        Load data from a .causal knowledge graph file.

        Args:
            file_path: Path to the .causal file
            query: Optional search query to filter triplets
            limit: Maximum number of triplets to return
            **load_kwargs: Additional arguments passed to search

        Returns:
            List of Document objects, each containing a triplet
        """
        from dotcausal import CausalReader as DotCausalReader

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Causal file not found: {file_path}")

        reader = DotCausalReader(str(path))

        # Get triplets - either search or get all
        if query:
            results = reader.search(
                query=query,
                limit=limit or 100,
                **load_kwargs,
            )
        else:
            # Get all triplets
            stats = reader.get_stats()
            results = []

            # Get explicit triplets
            explicit = reader.search("", limit=stats.get("explicit_triplets", 1000))
            results.extend(explicit)

            # Get inferred if requested
            if self.include_inferred:
                inferred = reader.search(
                    "", limit=stats.get("inferred_triplets", 1000)
                )
                for r in inferred:
                    if r.get("is_inferred") and r not in results:
                        results.append(r)

        # Filter and convert to Documents
        documents = []
        for r in results:
            # Apply filters
            if r.get("confidence", 1.0) < self.min_confidence:
                continue
            if not self.include_inferred and r.get("is_inferred", False):
                continue

            # Format content
            tag = "[INFERRED]" if r.get("is_inferred") else "[EXPLICIT]"
            content = (
                f"{tag} {r['trigger']} → {r['mechanism']} → {r['outcome']}"
            )

            # Build metadata
            metadata: Dict[str, Any] = {
                "file_path": str(file_path),
                "trigger": r["trigger"],
                "mechanism": r["mechanism"],
                "outcome": r["outcome"],
                "confidence": r.get("confidence", 1.0),
                "is_inferred": r.get("is_inferred", False),
                "source": r.get("source", ""),
                "provenance": r.get("provenance", []),
            }

            documents.append(
                Document(
                    text=content,
                    metadata=metadata,
                )
            )

            if limit and len(documents) >= limit:
                break

        return documents

    def get_stats(self, file_path: str) -> Dict[str, Any]:
        """
        Get statistics about a .causal knowledge graph.

        Args:
            file_path: Path to the .causal file

        Returns:
            Dictionary with triplet counts and amplification stats
        """
        from dotcausal import CausalReader as DotCausalReader

        reader = DotCausalReader(file_path)
        return reader.get_stats()
