import json
from typing import Any, Optional
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms.llm import LLM
from llama_index.core.response.schema import Response

VERIFICATION_PROMPT = """\
You are an adversarial epistemic fact-checking system.

Your job is to verify whether the DRAFT RESPONSE is fully supported by the SOURCE CONTEXT.

Rules:
- Only use the SOURCE CONTEXT as ground truth.
- Do NOT assume external knowledge.
- Evaluate each claim for support.

SOURCE CONTEXT:
---------------------
{context}
---------------------

USER QUERY:
{query}

DRAFT RESPONSE:
{response}

Analyze the draft response strictly against the context. Return ONLY valid JSON in this exact format:
{{
  "verdict": "PASS" or "FAIL",
  "confidence": <float between 0.0 and 1.0>,
  "explanation": "<brief reasoning for your verdict>"
}}
"""

class VerificationQueryEngine(CustomQueryEngine):
    """
    A post-RAG verification guardrail that wraps an existing query engine.
    It takes the drafted response and forces an adversarial LLM to cross-reference 
    it against the retrieved source nodes, generating a confidence score and verdict.
    """
    
    base_query_engine: BaseQueryEngine
    llm: LLM
    strict_mode: bool = False
    max_context_chars: int = 12000

    def _truncate_context(self, source_nodes: Any) -> str:
        """Safely truncate the context to prevent judge-LLM context window overflow."""
        context_chunks = []
        total_chars = 0

        for node_with_score in source_nodes:
            content = node_with_score.node.get_content()
            if total_chars + len(content) > self.max_context_chars:
                break
            context_chunks.append(content)
            total_chars += len(content)

        return "\n\n".join(context_chunks)

    def custom_query(self, query_str: str) -> Response:
        # Step 1: Execute the base query engine to get the drafted response
        response = self.base_query_engine.query(query_str)
        
        # If no source nodes were retrieved, verification defaults to false
        if not response.source_nodes:
            response.metadata = response.metadata or {}
            response.metadata["is_verified"] = False
            response.metadata["verification_reason"] = "No source nodes retrieved"
            if self.strict_mode:
                response.response = "The system could not retrieve any verifiable context to answer this query."
            return response
            
        # Step 2: Assemble bounded context
        context_str = self._truncate_context(response.source_nodes)
        
        # Step 3: Adversarial Verification Call
        prompt = VERIFICATION_PROMPT.format(
            context=context_str,
            query=query_str,
            response=response.response
        )
        
        raw_output = self.llm.complete(prompt).text.strip()
        
        # Step 4: Safe JSON Parsing
        try:
            # Strip markdown code blocks if the LLM hallucinated them
            clean_json = raw_output.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_json)
            verdict = result.get("verdict", "FAIL")
            confidence = float(result.get("confidence", 0.0))
            explanation = result.get("explanation", "No explanation provided.")
        except Exception:
            verdict = "FAIL"
            confidence = 0.0
            explanation = "Failed to parse structured JSON from verification LLM."

        is_pass = (verdict.upper() == "PASS")
        
        # Step 5: Metadata Injection
        response.metadata = response.metadata or {}
        response.metadata["verification"] = {
            "verdict": verdict,
            "confidence": confidence,
            "explanation": explanation,
            "raw_output": raw_output,
        }
        response.metadata["is_verified"] = is_pass
        
        # Step 6: Non-destructive Strict Mode Handling
        if self.strict_mode and not is_pass:
            response.metadata["strict_mode_blocked"] = True
            response.metadata["original_draft_response"] = response.response
            response.response = (
                "The initial response failed epistemic verification against the retrieved sources. "
                "The generated claims could not be conclusively substantiated."
            )
            
        return response
