from llama_index.core.tools.tool_spec.base import BaseToolSpec
import requests

class GlobalCheckToolSpec(BaseToolSpec):
    """GlobalCheck tool spec.
    
    A deterministic trade compliance verifier for AI agents.
    """

    spec_functions = ["check_compliance"]

    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def check_compliance(self, product_description: str, origin_country: str, destination_country: str) -> str:
        """
        CRITICAL GUARDRAIL: Automatically verify international trade compliance 
        for e-commerce or logistics shipments. Use this BEFORE performing any 
        purchase integrations to guarantee legal constraints and treaties.
        
        Args:
            product_description (str): Natural language product description
            origin_country (str): ISO 2-letter country code of origin
            destination_country (str): ISO 2-letter country code of destination
            
        Returns:
            str: A formatted compliance verdict string including the assigned HS code and legal reasoning.
        """
        url = "https://api.globalcheck.ai/v1/check-compliance"
        payload = {
            "product_description": product_description,
            "origin_country": origin_country,
            "destination_country": destination_country
        }
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["X-RapidAPI-Key"] = self.api_key
            headers["X-RapidAPI-Host"] = "globalcheck.p.rapidapi.com"

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            return (
                f"------- GLOBALCHECK COMPLIANCE VERDICT -------\n"
                f"STATUS: {data.get('status')}\n"
                f"HS CODE GENERATED: {data.get('hs_code_assigned')}\n"
                f"LEGAL REASONING: {data.get('reason')}\n"
                f"EVIDENCE: {data.get('evidence', 'N/A')}\n"
                f"----------------------------------------------\n"
            )
        except Exception as e:
            return f"Error verifying compliance: {str(e)}"
