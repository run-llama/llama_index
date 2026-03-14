"""HADS (Human-AI Document Standard) reader for LlamaIndex."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

BLOCK_PATTERN = re.compile(
    r"\*\*\[(SPEC|NOTE|BUG[^\]]*|\?)\]\*\*\n((?:(?!\*\*\[).*\n?)*)",
    re.MULTILINE,
)
MANIFEST_PATTERN = re.compile(
    r"##\s+AI READING INSTRUCTION\s*\n(.*?)(?=\n##|\Z)", re.DOTALL
)
HEADER_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


class HADSReader(BaseReader):
    """Reader for Human-AI Document Standard (HADS) files.

    HADS is a lightweight convention for AI-optimized technical documentation.
    Files contain tagged blocks like **[SPEC]**, **[NOTE]**, **[BUG title]**,
    and **[?]** that allow selective loading to reduce context usage by ~70%.

    Args:
        block_types: Block type prefixes to include. Defaults to ["SPEC"].
            Use ["SPEC", "NOTE", "BUG", "?"] to include all blocks.
        include_section_headers: Whether to prepend H1/H2/H3 headers to each
            block for context. Defaults to True.

    Example:
        .. code-block:: python

            from llama_index.readers.hads import HADSReader

            reader = HADSReader(block_types=["SPEC", "NOTE"])
            documents = reader.load_data(Path("architecture.hads.md"))
    """

    def __init__(
        self,
        block_types: Optional[List[str]] = None,
        include_section_headers: bool = True,
    ) -> None:
        self.block_types = block_types if block_types is not None else ["SPEC"]
        self.include_section_headers = include_section_headers

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict[str, Any]] = None,
        fs: Optional[Any] = None,
    ) -> List[Document]:
        """Load HADS blocks from a file.

        Args:
            file: Path to the HADS file.
            extra_info: Optional extra metadata to merge.
            fs: Optional fsspec filesystem. If provided, used instead of
                local filesystem.

        Returns:
            List of Documents, one per matching block.
        """
        if fs is not None:
            with fs.open(str(file), encoding="utf-8") as f:
                content = f.read()
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
        else:
            try:
                content = Path(file).read_text(encoding="utf-8")
            except FileNotFoundError as e:
                raise ValueError(f"File not found: {file}") from e

        blocks = self._extract_blocks(content)
        manifest = self._extract_manifest(content)

        metadata: Dict[str, Any] = {
            "source": str(file),
            "hads": True,
            "block_types": self.block_types,
            "blocks_found": len(blocks),
        }
        if manifest:
            metadata["manifest"] = manifest
        if extra_info:
            metadata.update(extra_info)

        if not blocks:
            return [Document(text="", metadata=metadata)]

        return [
            Document(
                text=block_text,
                metadata={**metadata, "block_tag": block_tag},
            )
            for block_tag, block_text in blocks
        ]

    def _should_include(self, tag: str) -> bool:
        """Return True if tag matches any requested block_type prefix."""
        tag_upper = tag.upper()
        return any(tag_upper.startswith(bt.upper()) for bt in self.block_types)

    def _extract_blocks(self, content: str) -> List[tuple]:
        """Extract matching tagged blocks, optionally with section headers."""
        blocks = []
        current_headers: Dict[int, str] = {}

        lines = content.splitlines(keepends=True)
        i = 0
        while i < len(lines):
            line = lines[i]

            # Track section headers for context
            hm = HEADER_PATTERN.match(line.rstrip())
            if hm:
                level = len(hm.group(1))
                current_headers[level] = hm.group(2).strip()
                # Clear lower-level headers when a higher-level one appears
                for lvl in list(current_headers):
                    if lvl > level:
                        del current_headers[lvl]
                i += 1
                continue

            # Check for HADS block tag
            tag_match = re.match(r"\*\*\[([^\]]+)\]\*\*", line.strip())
            if tag_match:
                tag = tag_match.group(1)
                if self._should_include(tag):
                    # Collect block body until next tag or EOF
                    body_lines = []
                    i += 1
                    while i < len(lines):
                        if re.match(r"\*\*\[([^\]]+)\]\*\*", lines[i].strip()):
                            break
                        body_lines.append(lines[i])
                        i += 1

                    body = "".join(body_lines).strip()

                    if self.include_section_headers and current_headers:
                        header_ctx = " > ".join(
                            current_headers[lvl]
                            for lvl in sorted(current_headers)
                        )
                        text = f"[{tag}] ({header_ctx})\n{body}"
                    else:
                        text = f"[{tag}]\n{body}"

                    blocks.append((tag, text))
                    continue
            i += 1

        return blocks

    def _extract_manifest(self, content: str) -> Optional[str]:
        """Extract the AI READING INSTRUCTION manifest if present."""
        m = MANIFEST_PATTERN.search(content)
        return m.group(1).strip() if m else None
