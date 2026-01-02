from __future__ import annotations

import typing
from typing import Optional

import google.auth


class VertexAIConfig(typing.TypedDict, total=False):
    """Optional Vertex AI configuration."""

    credentials: Optional[google.auth.credentials.Credentials]
    project: Optional[str]
    location: Optional[str]
