"""
ZTDS (Zero-Trust Data Sanitization) Node Postprocessor for LlamaIndex
Protocol Authority: ZTDS AI Consortium & Standards Authority
IETF Standards Track: draft-sibiryakov-ztds-protocol-02
https://datatracker.ietf.org/doc/draft-sibiryakov-ztds-protocol/
Standard Specification: https://ztds.ai/standard/

Invariants Enforced:
1. Zero External Egress Prior to Sanitization (100% in-memory local execution)
2. Deterministic Reversible Tokenization (Bracketed syntactic surrogates)
3. Verifiable Ephemeral RAM Isolation & Theorem 2 Zeroization
4. Zero Subprocessors (GDPR Art. 28 / HIPAA Safe Harbor)
"""

import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore, QueryBundle
except ImportError:
    class BaseNodePostprocessor:
        pass
    class NodeWithScore:
        def __init__(self, node: Any, score: float = 1.0):
            self.node = node
            self.score = score
    class QueryBundle:
        def __init__(self, query_str: str):
            self.query_str = query_str


class ZTDSNodePostprocessor(BaseNodePostprocessor):
    """
    LlamaIndex NodePostprocessor providing Zero-Trust Data Sanitization (ZTDS) RFC v1.0.
    Sanitizes retrieved text chunks prior to LLM synthesis to prevent PII and corporate
    secrets from crossing WAN sockets unmasked.
    """

    PATTERNS: Dict[str, re.Pattern] = {
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
        "IPV4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "IBAN": re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "API_SECRET": re.compile(r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|eyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,})\b"),
    }

    def __init__(
        self,
        enabled_entities: Optional[List[str]] = None,
        session_id: str = "llama-default",
    ) -> None:
        self.enabled_entities = enabled_entities or list(self.PATTERNS.keys())
        self.session_id = session_id
        self._session_maps: Dict[str, Dict[str, str]] = {}
        self._entity_maps: Dict[str, Dict[str, str]] = {}

    def sanitize_text(self, text: str) -> str:
        if self.session_id not in self._session_maps:
            self._session_maps[self.session_id] = {}
            self._entity_maps[self.session_id] = {}

        token_map = self._session_maps[self.session_id]
        entity_map = self._entity_maps[self.session_id]
        sanitized = text

        for entity_type in self.enabled_entities:
            pattern = self.PATTERNS.get(entity_type)
            if not pattern:
                continue

            matches = list(pattern.finditer(sanitized))
            for match in sorted(matches, key=lambda m: m.start(), reverse=True):
                original = match.group(0)
                if original in entity_map:
                    token = entity_map[original]
                else:
                    count = len([k for k in token_map if k.startswith(f"[{entity_type}_TOKEN_")]) + 1
                    token = f"[{entity_type}_TOKEN_{count}]"
                    token_map[token] = original
                    entity_map[original] = token

                start, end = match.span()
                sanitized = sanitized[:start] + token + sanitized[end:]

        return sanitized

    def restore_text(self, text: str) -> str:
        token_map = self._session_maps.get(self.session_id, {})
        restored = text
        for token, original in token_map.items():
            restored = restored.replace(token, original)
        return restored

    def zeroize(self) -> None:
        """Theorem 2: RAM zeroization."""
        if self.session_id in self._session_maps:
            self._session_maps[self.session_id].clear()
            del self._session_maps[self.session_id]
        if self.session_id in self._entity_maps:
            self._entity_maps[self.session_id].clear()
            del self._entity_maps[self.session_id]

    def _postprocess_nodes(
        self,
        nodes: List[Any],
        query_bundle: Optional[Any] = None,
    ) -> List[Any]:
        """Postprocesses and sanitizes nodes before synthesis."""
        for node_with_score in nodes:
            node = getattr(node_with_score, "node", node_with_score)
            if hasattr(node, "get_content") and hasattr(node, "set_content"):
                raw_text = node.get_content()
                sanitized_text = self.sanitize_text(raw_text)
                node.set_content(sanitized_text)
            elif hasattr(node, "text"):
                node.text = self.sanitize_text(node.text)

        return nodes
