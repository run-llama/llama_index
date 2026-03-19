"""Snowflake Cortex Embedding integration for LlamaIndex."""

import os
import warnings
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CallbackManager

from llama_index.embeddings.cortex.utils import (
    generate_sf_jwt,
    get_default_spcs_token,
    get_spcs_base_url,
    is_spcs_environment,
)

# https://docs.snowflake.com/en/user-guide/snowflake-cortex/aisql
EMBED_MODELS: Dict[str, Dict[str, Any]] = {
    # 768-dim models
    "snowflake-arctic-embed-m-v1.5": {"dimensions": 768, "max_tokens": 512},
    "snowflake-arctic-embed-m": {"dimensions": 768, "max_tokens": 512},
    "e5-base-v2": {"dimensions": 768, "max_tokens": 512},
    # 1024-dim models
    "snowflake-arctic-embed-l-v2.0": {"dimensions": 1024, "max_tokens": 512},
    "snowflake-arctic-embed-l-v2.0-8k": {
        "dimensions": 1024,
        "max_tokens": 8192,
    },
    "nv-embed-qa-4": {"dimensions": 1024, "max_tokens": 512},
    "multilingual-e5-large": {"dimensions": 1024, "max_tokens": 512},
    "voyage-multilingual-2": {"dimensions": 1024, "max_tokens": 32000},
}

DEFAULT_MODEL = "snowflake-arctic-embed-m-v1.5"
DEFAULT_EMBED_BATCH_SIZE = 32


class CortexEmbedding(BaseEmbedding):
    """
    Snowflake Cortex Embedding model.

    Generates text embeddings using the Snowflake Cortex REST API
    (POST /api/v2/cortex/inference:embed).

    Authentication methods (in order of precedence):
        1. Explicit: ``private_key_file``, ``jwt_token``, or ``session``
        2. Environment variable ``SNOWFLAKE_KEY_FILE``
        3. Auto-detected SPCS default OAuth token

    Examples:
        ``pip install llama-index-embeddings-cortex``

        ```python
        from llama_index.embeddings.cortex import CortexEmbedding

        embed = CortexEmbedding(
            model_name="snowflake-arctic-embed-m-v1.5",
            account="ORG-ACCOUNT",
            user="MY_USER",
            private_key_file="/path/to/rsa_key.p8",
        )

        embedding = embed.get_text_embedding("Hello world")
        ```

    """

    user: Optional[str] = Field(
        default=None,
        description="Snowflake user.",
    )
    account: Optional[str] = Field(
        default=None,
        description=(
            "Fully qualified Snowflake account specified as <ORG_ID>-<ACCOUNT_ID>."
        ),
    )
    private_key_file: Optional[str] = Field(
        default=None,
        description="Filepath to Snowflake private key file.",
    )
    jwt_token: Optional[str] = Field(
        default=None,
        description="JWT token string or filepath.",
    )
    session: Optional[Any] = Field(
        default=None,
        description="Snowpark Session object.",
    )

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        user: Optional[str] = None,
        account: Optional[str] = None,
        private_key_file: Optional[str] = None,
        jwt_token: Optional[str] = None,
        session: Optional[Any] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )

        # Resolve user / account from params or env
        self.user = user or os.environ.get("SNOWFLAKE_USERNAME")
        self.account = account or os.environ.get("SNOWFLAKE_ACCOUNT")

        # Authentication – same precedence as llama-index-llms-cortex
        is_in_spcs = is_spcs_environment()
        env_key_file = os.environ.get("SNOWFLAKE_KEY_FILE")

        if private_key_file:
            self.private_key_file = private_key_file
        elif jwt_token:
            if os.path.isfile(jwt_token):
                with open(jwt_token) as fp:
                    self.jwt_token = fp.read()
            else:
                self.jwt_token = jwt_token
        elif session:
            self.session = session
        elif env_key_file and not is_in_spcs:
            self.private_key_file = env_key_file
        elif is_in_spcs:
            self.jwt_token = get_default_spcs_token()
        else:
            raise ValueError(
                "Authentication required. Provide one of: "
                "private_key_file, jwt_token, session, "
                "set SNOWFLAKE_KEY_FILE env var, "
                "or run in an SPCS environment."
            )

        if is_in_spcs and self.session:
            warnings.warn(
                "SPCS environment detected. If using the default auth "
                "token, do NOT set 'user' and 'role' parameters or "
                "your auth may be rejected."
            )

    # -- Auth helpers ---------------------------------------------------------

    def _generate_auth_header(self) -> str:
        if self.jwt_token:
            return f"Bearer {self.jwt_token}"
        elif self.session:
            return f'Snowflake Token="{self.session.connection.rest.token}"'
        elif self.private_key_file:
            return (
                f"Bearer "
                f"{generate_sf_jwt(self.account, self.user, self.private_key_file)}"
            )
        else:
            raise ValueError("Cortex Embedding: no authentication method set.")

    @property
    def _api_base_url(self) -> str:
        if is_spcs_environment():
            return "https://" + get_spcs_base_url()
        return f"https://{self.account}.snowflakecomputing.com"

    @property
    def _embed_endpoint(self) -> str:
        return self._api_base_url + "/api/v2/cortex/inference:embed"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": self._generate_auth_header(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # -- Sync helpers ---------------------------------------------------------

    def _call_embed_api(self, texts: List[str]) -> List[Embedding]:
        """Call the Cortex embed endpoint synchronously."""
        payload = {"model": self.model_name, "text": texts}
        resp = requests.post(
            self._embed_endpoint,
            headers=self._build_headers(),
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        # Response data[i].embedding is a nested array — flatten
        return [item["embedding"][0] for item in data["data"]]

    # -- Async helpers --------------------------------------------------------

    async def _acall_embed_api(self, texts: List[str]) -> List[Embedding]:
        """Call the Cortex embed endpoint asynchronously."""
        payload = {"model": self.model_name, "text": texts}
        async with aiohttp.ClientSession() as http_session:
            async with http_session.post(
                self._embed_endpoint,
                headers=self._build_headers(),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        return [item["embedding"][0] for item in data["data"]]

    # -- Required abstract methods --------------------------------------------

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._call_embed_api([query])[0]

    async def _aget_query_embedding(self, query: str) -> Embedding:
        results = await self._acall_embed_api([query])
        return results[0]

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._call_embed_api([text])[0]

    # -- Optional batch methods -----------------------------------------------

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """Batch embed — sends all texts in one API call."""
        return self._call_embed_api(texts)

    async def _aget_text_embedding(self, text: str) -> Embedding:
        results = await self._acall_embed_api([text])
        return results[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """Async batch embed — sends all texts in one API call."""
        return await self._acall_embed_api(texts)
