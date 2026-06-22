"""BGPT Tool Spec for LlamaIndex."""

import logging
from typing import Any, Dict, List, Optional

import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec

logger = logging.getLogger(__name__)

class BGPTToolSpec(BaseToolSpec):
    """BGPT Tool Spec for retrieving structured scientific literature evidence.
    
    This tool interacts with the BGPT API to fetch comprehensive, study-level 
    evidence including methods, limitations, quality scores, and falsifiability.
    """

    spec_functions = ["search_literature"]
    DEFAULT_BASE_URL = "https://bgpt.pro/api/mcp-search"

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize the BGPT Tool Spec.

        Args:
            api_key (Optional[str]): The BGPT API key. Only required when 
                                     requesting more than 50 results.
        """
        self.api_key = api_key
        self.base_url = self.DEFAULT_BASE_URL

    def search_literature(
        self,
        query: str,
        num_results: int = 5,
        days_back: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for scientific literature and structured evidence using BGPT.
        
        Use this tool when you need deep, study-level evidence such as methods, 
        limitations, sample sizes, and quality scores, rather than just abstracts.

        Args:
            query (str): The specific research question or topic to search for 
                         (e.g., 'effects of sleep deprivation on memory').
            num_results (int): The number of top results to return. Defaults to 5.
            days_back (Optional[int]): Filter results to only include papers published 
                                       within the last N days.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing structured 
                                  fields of a scientific paper, or an error dictionary.
        """
        payload: Dict[str, Any] = {
            "query": query,
            "num_results": num_results,
        }

        if days_back is not None:
            payload["days_back"] = days_back
            
        if self.api_key is not None:
            payload["api_key"] = self.api_key

        try:
            response = requests.post(self.base_url, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data.get("results", [])

        except requests.exceptions.RequestException as e:
            logger.error(f"Network or HTTP error fetching data from BGPT API: {e}")
            return [{"error": f"API request failed: {str(e)}"}]
        except ValueError as e:
            logger.error(f"Failed to parse JSON response from BGPT API: {e}")
            return [{"error": "Invalid JSON response received from API."}]