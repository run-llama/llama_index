import re
from typing import Any, Callable, Dict, List, Optional

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms import LLM


class ProviderConfig:
    """Configuration for an AI provider."""

    def __init__(
        self,
        name: str,
        regex: str,
        transform_endpoint: Callable[[str], str],
    ):
        self.name = name
        self.regex = re.compile(regex)
        self.transform_endpoint = transform_endpoint


def transform_azure_endpoint(url: str) -> str:
    """Transform Azure OpenAI endpoint URL."""
    match = re.match(
        r"^https://(?P<resource>[^.]+)\.openai\.azure\.com/openai/deployments/(?P<deployment>[^/]+)/(?P<rest>.*)$",
        url,
    )
    if not match:
        return url

    groups = match.groupdict()
    resource = groups.get("resource")
    deployment = groups.get("deployment")
    rest = groups.get("rest")

    if not all([resource, deployment, rest]):
        raise ValueError("Failed to parse Azure OpenAI endpoint URL.")

    return f"{resource}/{deployment}/{rest}"


# Define supported providers - matching the reference implementation
PROVIDERS = [
    ProviderConfig(
        name="openai",
        regex=r"^https://api\.openai\.com/",
        transform_endpoint=lambda url: url.replace("https://api.openai.com/", ""),
    ),
    ProviderConfig(
        name="anthropic",
        regex=r"^https://api\.anthropic\.com/",
        transform_endpoint=lambda url: url.replace("https://api.anthropic.com/", ""),
    ),
    ProviderConfig(
        name="google-ai-studio",
        regex=r"^https://generativelanguage\.googleapis\.com/",
        transform_endpoint=lambda url: url.replace(
            "https://generativelanguage.googleapis.com/", ""
        ),
    ),
    ProviderConfig(
        name="mistral",
        regex=r"^https://api\.mistral\.ai/",
        transform_endpoint=lambda url: url.replace("https://api.mistral.ai/", ""),
    ),
    ProviderConfig(
        name="groq",
        regex=r"^https://api\.groq\.com/openai/v1/",
        transform_endpoint=lambda url: url.replace(
            "https://api.groq.com/openai/v1/", ""
        ),
    ),
    ProviderConfig(
        name="deepseek",
        regex=r"^https://api\.deepseek\.com/",
        transform_endpoint=lambda url: url.replace("https://api.deepseek.com/", ""),
    ),
    ProviderConfig(
        name="perplexity-ai",
        regex=r"^https://api\.perplexity\.ai/",
        transform_endpoint=lambda url: url.replace("https://api.perplexity.ai/", ""),
    ),
    ProviderConfig(
        name="replicate",
        regex=r"^https://api\.replicate\.com/",
        transform_endpoint=lambda url: url.replace("https://api.replicate.com/", ""),
    ),
    ProviderConfig(
        name="grok",
        regex=r"^https://api\.x\.ai/",
        transform_endpoint=lambda url: url.replace("https://api.x.ai/", ""),
    ),
    ProviderConfig(
        name="azure-openai",
        regex=r"^https://(?P<resource>[^.]+)\.openai\.azure\.com/openai/deployments/(?P<deployment>[^/]+)/(?P<rest>.*)$",
        transform_endpoint=transform_azure_endpoint,
    ),
]


def messages_to_dict(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """Convert ChatMessage objects to dictionary format."""
    result = []
    for msg in messages:
        msg_dict = {
            "role": msg.role.value,
            "content": msg.content,
        }
        if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
            msg_dict.update(msg.additional_kwargs)
        result.append(msg_dict)
    return result


def get_provider_config(llm: LLM) -> Optional[ProviderConfig]:
    """Get the provider configuration for an LLM."""
    # Try to get the base URL from the LLM
    base_url = None

    # Check different possible attributes
    if hasattr(llm, "api_base") and llm.api_base:
        base_url = llm.api_base
    elif hasattr(llm, "base_url") and llm.base_url:
        base_url = llm.base_url
    elif hasattr(llm, "_client") and llm._client and hasattr(llm._client, "base_url"):
        base_url = llm._client.base_url

    # Ensure base_url is a string if it exists
    if base_url is not None:
        # Convert to string if it's not already
        if not isinstance(base_url, str):
            base_url = str(base_url)

        # Match by base URL
        for provider in PROVIDERS:
            if provider.regex.match(base_url):
                return provider

    # If we can't find a base URL or no match, try to match by class name
    class_name = type(llm).__name__.lower()
    if "openai" in class_name:
        return next((p for p in PROVIDERS if p.name == "openai"), None)
    elif "anthropic" in class_name:
        return next((p for p in PROVIDERS if p.name == "anthropic"), None)
    elif "azure" in class_name:
        return next((p for p in PROVIDERS if p.name == "azure-openai"), None)
    elif "google" in class_name:
        return next((p for p in PROVIDERS if p.name == "google-ai-studio"), None)
    elif "mistral" in class_name:
        return next((p for p in PROVIDERS if p.name == "mistral"), None)
    elif "groq" in class_name:
        return next((p for p in PROVIDERS if p.name == "groq"), None)
    elif "deepseek" in class_name:
        return next((p for p in PROVIDERS if p.name == "deepseek"), None)
    elif "perplexity" in class_name:
        return next((p for p in PROVIDERS if p.name == "perplexity-ai"), None)
    elif "replicate" in class_name:
        return next((p for p in PROVIDERS if p.name == "replicate"), None)
    elif "grok" in class_name:
        return next((p for p in PROVIDERS if p.name == "grok"), None)
    else:
        return None
