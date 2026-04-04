from typing import Any, List, Optional
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms.llm import LLM
from llama_index.core.response.schema import Response

# The strict adversarial prompt used to audit the base generator
VERIFICATION_PROMPT = """\
You are an adversarial epistemic fact-checker. 
A draft response to a user's query has been generated based exclusively on the provided source context.
Your goal is to actively challenge the response. Parse the response and determine if any claim is hallucinated, contradicted by the context, or unmentioned in the context.

SOURCE CONTEXT: 
---------------------
{context}
---------------------

USER QUERY: {query}
DRAFT RESPONSE: {response}

Analyze the draft response strictly against the context. You must output your final verdict in the exact following format:
VERDICT: [PASS/FAIL]
EXPLANATION: [Your concise reasoning]
"""

class VerificationQueryEngine(CustomQueryEngine):
    """
    A post-RAG verification guardrail that wraps an existing query engine.
    It takes the drafted response and forces an adversarial LLM to cross-reference 
    it against the retrieved source nodes.
    """
    # Define standard Pydantic fields
    base_query_engine: BaseQueryEngine
    llm: LLM
    strict_mode: bool = False

    def custom_query(self, query_str: str) -> Response:
        # Step 1: Execute the base query engine to get the drafted response
        response = self.base_query_engine.query(query_str)
        
        # If no source nodes were retrieved, there is nothing to verify against.
        if not response.source_nodes:
            if self.strict_mode:
                response.response = "The system could not retrieve any verifiable context to answer this query."
            return response
            
        # Step 2: Context Assembly (Linear concatenation for simplicity in v1)
        context_str = "\n\n".join([n.node.get_content() for n in response.source_nodes])
        
        # Step 3: Adversarial Verification
        prompt = VERIFICATION_PROMPT.format(
            context=context_str,
            query=query_str,
            response=response.response
        )
        
        verification_result = self.llm.complete(prompt)
        verdict_text = verification_result.text.strip()
        
        # Step 4: Metadata Injection
        response.metadata = response.metadata or {}
        response.metadata["verification_raw_output"] = verdict_text
        
        # Determine pass/fail
        is_pass = "VERDICT: PASS" in verdict_text.upper()
        response.metadata["is_verified"] = is_pass
        
        # Step 5: Output Modification
        if self.strict_mode and not is_pass:
            response.response = (
                "The initial response failed epistemic verification against the retrieved sources. "
                "The generated claims could not be conclusively substantiated."
            )
            
        return response
