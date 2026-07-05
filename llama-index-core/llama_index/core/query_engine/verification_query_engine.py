from typing import Any
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms.llm import LLM
from llama_index.core.response.schema import Response
import json


VERIFICATION_PROMPT = """\
You are an adversarial fact-checking system.

Your job is to verify whether the DRAFT RESPONSE is fully supported by the SOURCE CONTEXT.

Rules:
- Only use SOURCE CONTEXT as ground truth
- Do NOT assume external knowledge
- Evaluate each claim for support

SOURCE CONTEXT:
---------------------
{context}
---------------------

USER QUERY:
{query}

DRAFT RESPONSE:
{response}

Return ONLY valid JSON in this format:
{
  "verdict": "PASS" or "FAIL",
  "confidence": float between 0 and 1,
  "explanation": "brief reasoning"
}
"""


class VerificationQueryEngine(CustomQueryEngine):
    """
    Post-RAG verification wrapper that validates responses against retrieved sources.
    """

    base_query_engine: BaseQueryEngine
    llm: LLM
    strict_mode: bool = False
    max_context_chars: int = 12000

    def _truncate_context(self, source_nodes) -> str:
        context = []
        total = 0

        for n in source_nodes:
            content = n.node.get_content()
            if total + len(content) > self.max_context_chars:
                break
            context.append(content)
            total += len(content)

        return "\n\n".join(context)

    def custom_query(self, query_str: str) -> Response:
        # Step 1: base response
        response = self.base_query_engine.query(query_str)

        if not response.source_nodes:
            response.metadata = response.metadata or {}
            response.metadata["is_verified"] = False
            response.metadata["verification_reason"] = "No source nodes retrieved"
            return response

        # Step 2: bounded context
        context_str = self._truncate_context(response.source_nodes)

        # Step 3: verification call
        prompt = VERIFICATION_PROMPT.format(
            context=context_str,
            query=query_str,
            response=response.response
        )

        raw = self.llm.complete(prompt).text.strip()

        # Step 4: safe JSON parsing
        try:
            result = json.loads(raw)
            verdict = result.get("verdict", "FAIL")
            confidence = float(result.get("confidence", 0.0))
            explanation = result.get("explanation", "")
        except Exception:
            verdict = "FAIL"
            confidence = 0.0
            explanation = "Failed to parse verification output"

        is_pass = verdict.upper() == "PASS"

        # Step 5: metadata injection (safe + structured)
        response.metadata = response.metadata or {}
        response.metadata["verification"] = {
            "verdict": verdict,
            "confidence": confidence,
            "explanation": explanation,
            "raw_output": raw,
        }
        response.metadata["is_verified"] = is_pass

        # Step 6: strict mode (non-destructive)
        if self.strict_mode and not is_pass:
            response.metadata["strict_mode_blocked"] = True
            response.metadata["original_response"] = response.response
            response.metadata["warning"] = (
                "Response failed verification against source context"
            )

        return response
