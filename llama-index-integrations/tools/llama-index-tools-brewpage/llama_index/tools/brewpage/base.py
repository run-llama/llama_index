"""BrewPage Tool Spec for publishing and retrieving HTML/Markdown content."""

from typing import Optional

import requests

from llama_index.core.tools.tool_spec.base import BaseToolSpec


class BrewPageToolSpec(BaseToolSpec):
    """
    BrewPage tool spec for publishing and retrieving HTML/Markdown content.

    BrewPage is a simple HTML/Markdown/JSON/file hosting service with no auth required.
    Free tier supports up to 5 MB per resource with a 15-day default TTL.

    Example:
        ```python
        tool = BrewPageToolSpec()
        link = tool.publish_content("<h1>Hello</h1>", namespace="public")
        content = tool.get_content("public", link.split("/")[-1])
        ```

    """

    spec_functions = ["publish_content", "get_content"]
    base_url: str = "https://brewpage.app"

    def __init__(self, base_url: Optional[str] = None) -> None:
        """
        Initialize BrewPageToolSpec.

        Args:
            base_url: Optional base URL for BrewPage instance (default: https://brewpage.app)

        """
        if base_url:
            self.base_url = base_url

    def publish_content(
        self,
        content: str,
        namespace: str = "public",
        ttl_days: Optional[int] = None,
    ) -> str:
        """
        Publish HTML or Markdown content to BrewPage.

        Args:
            content: The HTML or Markdown content to publish (max 5 MB).
            namespace: The namespace to publish to (default: "public").
                Only "public" is recommended for sharing; private namespaces
                are for personal use only.
            ttl_days: Time-to-live in days (default: 15, max: 30).
                After TTL expires, the content is automatically deleted.

        Returns:
            str: A short URL link to access the published content.

        Raises:
            requests.RequestException: If the API request fails.

        """
        url = f"{self.base_url}/api/html"
        payload: dict = {
            "content": content,
            "namespace": namespace,
        }
        if ttl_days is not None:
            payload["ttl_days"] = ttl_days

        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()

        data = response.json()
        link: str = data.get("link", "")
        if not link:
            raise ValueError("No link returned from BrewPage API")
        return link

    def get_content(self, namespace: str, short_id: str) -> str:
        """
        Retrieve HTML or Markdown content from BrewPage.

        Args:
            namespace: The namespace where the content is stored.
            short_id: The short ID of the content (10-character identifier).

        Returns:
            str: The full HTML or Markdown content.

        Raises:
            requests.RequestException: If the API request fails (404 if not found).

        """
        url = f"{self.base_url}/api/html/{namespace}/{short_id}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()
        body: str = data.get("body", "")
        return body
