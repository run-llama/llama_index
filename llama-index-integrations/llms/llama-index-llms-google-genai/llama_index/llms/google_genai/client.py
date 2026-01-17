from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import google.genai
import google.genai.types as types

from llama_index.llms.google_genai.types import VertexAIConfig


class GenAIClientFactory:
    """Factory for creating a configured Google GenAI client."""

    @staticmethod
    def create(
        *,
        model: str,
        api_key: Optional[str],
        vertexai_config: Optional[VertexAIConfig] = None,
        http_options: Optional[types.HttpOptions] = None,
        debug_config: Optional[google.genai.client.DebugConfig] = None,
    ) -> Tuple[google.genai.Client, types.Model]:
        """
        Create a GenAI client and fetch model metadata.
        """
        config_params: Dict[str, Any] = {
            "api_key": api_key,
        }

        vertexai = (
            vertexai_config is not None
            or os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false") != "false"
        )

        project = (vertexai_config or {}).get("project") or os.getenv(
            "GOOGLE_CLOUD_PROJECT", None
        )
        location = (vertexai_config or {}).get("location") or os.getenv(
            "GOOGLE_CLOUD_LOCATION", None
        )

        if vertexai_config is not None:
            config_params.update(vertexai_config)
            config_params["api_key"] = None
            config_params["vertexai"] = True
        elif vertexai:
            config_params["project"] = project
            config_params["location"] = location
            config_params["api_key"] = None
            config_params["vertexai"] = True

        if http_options is not None:
            config_params["http_options"] = http_options

        if debug_config is not None:
            config_params["debug_config"] = debug_config

        client = google.genai.Client(**config_params)
        model_meta = client.models.get(model=model)
        return client, model_meta
