from types import MappingProxyType
from typing import Any, Dict

# Use these as kwargs for OpenAILike to connect to LocalAIs
DEFAULT_LOCALAI_PORT = 8080
LOCALAI_DEFAULTS: Dict[str, Any] = MappingProxyType(  # type: ignore[assignment]
    {
        "api_key": "localai_fake",
        "api_type": "localai_fake",
        "api_base": f"http://localhost:{DEFAULT_LOCALAI_PORT}/v1",
    }
)
