"""
APIVerve Tool Specification for LlamaIndex.

Provides access to 300+ utility APIs for AI agents including validation,
conversion, generation, analysis, and lookup tools.

Schemas are fetched from APIVerve at initialization and cached in memory.
"""

import os
from importlib.metadata import version as get_version
from typing import Any, Dict, List, Optional

import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec

# Package version - dynamically loaded from pyproject.toml
try:
    __version__ = get_version("llama-index-tools-apiverve")
except Exception:
    __version__ = "0.1.0"  # Fallback for development

# Schema source URL
SCHEMA_URL = "https://assets.apiverve.com/mcp-schemas.json"

# Module-level cache - schemas are fetched once per process
_schemas_cache: Optional[Dict[str, Any]] = None


def _load_schemas() -> Dict[str, Any]:
    """Load and cache API schemas from APIVerve."""
    global _schemas_cache

    if _schemas_cache is not None:
        return _schemas_cache

    try:
        response = requests.get(SCHEMA_URL, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not data.get("schemas"):
            raise ValueError("Response missing 'schemas' field")

        _schemas_cache = data.get("schemas", {})
        return _schemas_cache

    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Failed to fetch API schemas from {SCHEMA_URL}: {e}. "
            "APIVerve schema endpoint must be accessible."
        ) from e


class APIVerveToolSpec(BaseToolSpec):
    """
    Tool specification for accessing APIVerve APIs.

    APIVerve provides 300+ utility APIs including email validation, DNS lookup,
    IP geolocation, QR code generation, currency conversion, and more.

    Args:
        api_key: Your APIVerve API key. Get one at https://dashboard.apiverve.com
        base_url: Base URL for API calls. Defaults to https://api.apiverve.com/v1

    Example:
        >>> from llama_index.tools.apiverve import APIVerveToolSpec
        >>> from llama_index.agent.openai import OpenAIAgent
        >>>
        >>> # Initialize the tool spec
        >>> apiverve = APIVerveToolSpec(api_key="your-api-key")
        >>>
        >>> # Create an agent with APIVerve tools
        >>> agent = OpenAIAgent.from_tools(apiverve.to_tool_list())
        >>>
        >>> # Use the agent
        >>> response = agent.chat("Is test@example.com a valid email?")

    """

    spec_functions = [
        "call_api",
        "list_available_apis",
        "list_categories",
        "get_api_details",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.apiverve.com/v1",
    ) -> None:
        """Initialize the APIVerve tool specification."""
        self.api_key = api_key or os.environ.get("APIVERVE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Provide api_key parameter or set "
                "APIVERVE_API_KEY environment variable. "
                "Get your API key at https://dashboard.apiverve.com"
            )

        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "x-api-key": self.api_key,
                "Accept": "application/json",
                "User-Agent": f"llama-index-tools-apiverve/{__version__}",
            }
        )

        # Load API schemas (cached at module level)
        self._schemas = _load_schemas()

    def call_api(
        self,
        api_id: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Call any APIVerve API by its ID.

        This is the main tool for executing APIVerve API calls. Use list_available_apis()
        to discover available APIs and their parameters.

        Args:
            api_id: The API identifier (e.g., "emailvalidator", "dnslookup", "iplookup").
                   Use list_available_apis() to see all available API IDs.
            parameters: Dictionary of parameters to pass to the API.
                       Required parameters depend on the specific API.

        Returns:
            API response as a dictionary with 'status', 'error', and 'data' fields.

        Examples:
            - Validate email: call_api("emailvalidator", {"email": "test@example.com"})
            - DNS lookup: call_api("dnslookup", {"domain": "example.com"})
            - IP geolocation: call_api("iplookup", {"ip": "8.8.8.8"})
            - Generate QR code: call_api("qrcodegenerator", {"value": "https://example.com"})
            - Convert currency: call_api("currencyconverter", {"from": "USD", "to": "EUR", "amount": 100})

        """
        # Get schema for this API
        schema = self._schemas.get(api_id)
        if not schema:
            available = list(self._schemas.keys())[:10]
            raise ValueError(
                f"Unknown API: '{api_id}'. "
                f"Available APIs include: {', '.join(available)}... "
                f"Use list_available_apis() to see all {len(self._schemas)} APIs."
            )

        # Determine HTTP method
        methods = schema.get("methods", ["GET"])
        method = methods[0] if methods else "GET"

        # Make the API call
        url = f"{self.base_url}/{api_id}"
        params = parameters or {}

        try:
            if method.upper() == "POST":
                response = self._session.post(url, json=params, timeout=30)
            else:
                response = self._session.get(url, params=params, timeout=30)

            response.raise_for_status()
            data = response.json()
            return {"status": "ok", "error": None, "data": data}

        except requests.exceptions.HTTPError as e:
            error_msg = f"API request failed: {e}"
            try:
                error_response = e.response.json()
                if "error" in error_response:
                    error_msg = error_response["error"]
            except Exception:
                # Response body may not be valid JSON; use default error message
                pass
            return {"status": "error", "error": error_msg, "data": None}

        except requests.exceptions.RequestException as e:
            return {"status": "error", "error": f"Request failed: {e}", "data": None}

    def list_available_apis(
        self,
        category: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, str]]:
        """
        List available APIVerve APIs.

        Use this to discover what APIs are available and their descriptions.
        Then use call_api() to execute them.

        Args:
            category: Filter by category (e.g., "Validation", "Lookup", "Generation").
                     Use list_categories() to see all categories.
            search: Search term to filter APIs by name or description.
            limit: Maximum number of APIs to return. Default 50.

        Returns:
            List of dictionaries with 'id', 'title', 'description', and 'category'.

        """
        results = []

        for api_id, schema in self._schemas.items():
            # Filter by category
            if category and schema.get("category") != category:
                continue

            # Filter by search term
            if search:
                search_lower = search.lower()
                title = schema.get("title", "").lower()
                desc = schema.get("description", "").lower()
                if search_lower not in title and search_lower not in desc:
                    continue

            results.append(
                {
                    "id": api_id,
                    "title": schema.get("title", api_id),
                    "description": schema.get("description", ""),
                    "category": schema.get("category", "Other"),
                }
            )

            if len(results) >= limit:
                break

        return results

    def list_categories(self) -> List[str]:
        """
        List all available API categories.

        Use this to understand the types of APIs available, then use
        list_available_apis(category="...") to see APIs in a specific category.

        Returns:
            Sorted list of category names.

        """
        categories = set()
        for schema in self._schemas.values():
            cat = schema.get("category", "Other")
            categories.add(cat)
        return sorted(categories)

    def get_api_details(self, api_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific API including its parameters.

        Args:
            api_id: The API identifier.

        Returns:
            Dictionary with full API schema including parameters, or None if not found.

        """
        schema = self._schemas.get(api_id)
        if not schema:
            return None

        return {
            "id": schema.get("apiId"),
            "title": schema.get("title"),
            "description": schema.get("description"),
            "category": schema.get("category"),
            "methods": schema.get("methods", ["GET"]),
            "parameters": schema.get("parameters", []),
        }
