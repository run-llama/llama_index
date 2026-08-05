"""Ice Phi Shield API Guardrails Integration for LlamaIndex."""

import os
from typing import Any, Optional
import requests

from llama_index.core.bridge.pydantic import Field
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle


class IcePhiPromptGuard(BaseNodePostprocessor):
    """Ice Phi Prompt Shield Postprocessor calling api.icephi.com/shield."""

    api_key: str = Field(description="Ice Phi API Key")
    api_url: str = Field(
        default="https://api.icephi.com/shield",
        description="Endpoint URL for the Ice Phi Shield API"
    )
    timeout: float = Field(
        default=5.0,
        description="HTTP request timeout in seconds"
    )
    raise_on_blocked: bool = Field(
        default=True, 
        description="Whether to raise ValueError when prompt injection is detected."
    )

    @classmethod
    def class_name(cls) -> str:
        return "IcePhiPromptGuard"

    def __init__(
        self, 
        api_key: Optional[str] = None,
        api_url: str = "https://api.icephi.com/shield",
        timeout: float = 5.0,
        raise_on_blocked: bool = True,
        **kwargs: Any
    ) -> None:
        resolved_key = api_key or os.getenv("ICEPHI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "An Ice Phi API key must be provided via `api_key` or set in the "
                "ICEPHI_API_KEY environment variable."
            )

        super().__init__(
            api_key=resolved_key,
            api_url=api_url,
            timeout=timeout,
            raise_on_blocked=raise_on_blocked,
            **kwargs
        )

    def _verify_prompt_via_api(self, prompt: str) -> bool:
        """Sends the user prompt to api.icephi.com/shield."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {"prompt": prompt}

        try:
            response = requests.post(
                self.api_url, 
                json=payload, 
                headers=headers, 
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            # Checks for standard pass status (supports either 'is_clean' or 'pass')
            if "is_clean" in data:
                return data["is_clean"]
            return not data.get("blocked", False)

        except requests.RequestException as err:
            raise RuntimeError(f"Ice Phi Shield API request failed: {err}") from err

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        """Validates query_bundle against api.icephi.com/shield prior to node processing."""
        if query_bundle and query_bundle.query_str:
            is_clean = self._verify_prompt_via_api(query_bundle.query_str)
            if not is_clean:
                if self.raise_on_blocked:
                    raise ValueError("🚨 [Ice Phi Shield] Prompt Injection Detected!")
                return []
        return nodes
