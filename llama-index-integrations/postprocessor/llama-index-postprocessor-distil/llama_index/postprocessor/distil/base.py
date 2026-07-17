"""distil NodePostprocessor for LlamaIndex.

Compress each retrieved node's text with distil's reversible line digest before it
reaches the LLM synthesizer. distil keeps the head/tail and every salient line,
replacing the dropped middle with a single ``<< +N lines, handle=XXXXXXXX >>`` marker.

The compression is *reversible*: the original node text is written to distil's on-disk
handle store, so the exact bytes can be recovered later via distil's ``distil_expand``
MCP tool or :func:`distil.mcp_server.load_restore`. Best suited to long, structured node
content (logs, code, command/tool output); short nodes (<= head+tail+1 lines) are returned
unchanged.

This performs the *compression* only. Decision-equivalence between compressed and full
context is certified separately, offline, by ``distil bench`` / ``distil validate``.
"""

from __future__ import annotations

import re
from typing import List, Optional

from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore

from distil.compress.tier1 import digest
from distil.mcp_server import record_restore

# distil emits its recoverable handle inside the digest marker as ``handle=<8 hex>``.
_HANDLE_RE = re.compile(r"handle=([0-9a-f]{8})")


class DistilNodePostprocessor(BaseNodePostprocessor):
    """Reversibly compress retrieved node text with distil.

    Args:
        query_aware: when True (default), the query's terms are passed to distil as
            salience ``intent`` so lines naming what the query asks for are pinned.
            This only ever *widens* the set of lines kept, so it never drops an answer.
    """

    query_aware: bool = True

    @classmethod
    def class_name(cls) -> str:
        return "DistilNodePostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        intent: frozenset = frozenset()
        if self.query_aware and query_bundle is not None:
            intent = frozenset(query_bundle.query_str.lower().split())

        for nws in nodes:
            node = nws.node
            text = getattr(node, "text", None)
            if not isinstance(text, str) or not text:
                continue  # non-text / empty node — leave it untouched
            compressed, changed = digest(text, intent=intent)
            if not changed:
                continue  # too short to compress, or nothing dropped
            # Persist the original so its handle expands later (distil_expand / load_restore).
            for handle in set(_HANDLE_RE.findall(compressed)):
                record_restore(handle, text)
            node.text = compressed
        return nodes
