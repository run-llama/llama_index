"""Snowflake Cortex Search retriever integration for LlamaIndex."""

import os
import warnings
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from llama_index.retrievers.cortex_search.utils import (
    generate_sf_jwt,
    get_default_spcs_token,
    get_spcs_base_url,
    is_spcs_environment,
)


class CortexSearchRetriever(BaseRetriever):
    """
    Retriever for Snowflake Cortex Search Service.

    Queries a Cortex Search Service via the REST API
    (POST /api/v2/databases/{db}/schemas/{schema}/cortex-search-services/{name}:query)
    and returns results as LlamaIndex ``NodeWithScore`` objects.

    Authentication methods (in order of precedence):
        1. Explicit: ``private_key_file``, ``jwt_token``, or ``session``
        2. Environment variable ``SNOWFLAKE_KEY_FILE``
        3. Auto-detected SPCS default OAuth token

    Examples:
        ``pip install llama-index-retrievers-cortex-search``

        ```python
        from llama_index.retrievers.cortex_search import CortexSearchRetriever

        retriever = CortexSearchRetriever(
            service_name="my_search_svc",
            database="MY_DB",
            schema="MY_SCHEMA",
            search_column="content",
            columns=["content", "title", "url"],
            account="ORG_ID-ACCOUNT_ID",
            user="MY_USER",
            private_key_file="/path/to/rsa_key.p8",
        )

        nodes = retriever.retrieve("What is Snowflake?")
        for node in nodes:
            print(node.text, node.score)
        ```

    """

    def __init__(
        self,
        service_name: str,
        database: str,
        schema: str,
        search_column: str,
        columns: Optional[List[str]] = None,
        filter_spec: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        account: Optional[str] = None,
        user: Optional[str] = None,
        private_key_file: Optional[str] = None,
        jwt_token: Optional[str] = None,
        session: Optional[Any] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            callback_manager=callback_manager,
            verbose=verbose,
        )

        self.service_name = service_name
        self.database = database
        self.schema = schema
        self.search_column = search_column
        self.columns = columns or [search_column]
        self.filter_spec = filter_spec
        self.limit = limit

        # Ensure search_column is included in returned columns
        if self.search_column not in self.columns:
            self.columns = [self.search_column, *self.columns]

        # Auth setup — same pattern as llama-index-llms-cortex
        self.user = user or os.environ.get("SNOWFLAKE_USERNAME")
        self.account = account or os.environ.get("SNOWFLAKE_ACCOUNT")
        self.private_key_file: Optional[str] = None
        self.jwt_token: Optional[str] = None
        self.session: Optional[Any] = None

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
            raise ValueError("Cortex Search Retriever: no authentication method set.")

    @property
    def _api_base_url(self) -> str:
        if is_spcs_environment():
            return "https://" + get_spcs_base_url()
        return f"https://{self.account}.snowflakecomputing.com"

    @property
    def _search_endpoint(self) -> str:
        return (
            f"{self._api_base_url}/api/v2/databases/{self.database}"
            f"/schemas/{self.schema}"
            f"/cortex-search-services/{self.service_name}:query"
        )

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": self._generate_auth_header(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _build_request_body(self, query: str) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "query": query,
            "columns": self.columns,
            "limit": self.limit,
        }
        if self.filter_spec:
            body["filter"] = self.filter_spec
        return body

    # -- Result parsing -------------------------------------------------------

    def _parse_results(self, results: List[Dict[str, Any]]) -> List[NodeWithScore]:
        nodes: List[NodeWithScore] = []
        for result in results:
            text = result.get(self.search_column, "")
            # Build metadata from all other columns
            metadata = {k: v for k, v in result.items() if k != self.search_column}
            score = result.get("@search_score", 1.0)
            node = TextNode(text=str(text), metadata=metadata)
            nodes.append(NodeWithScore(node=node, score=float(score)))
        return nodes

    # -- Sync retrieval -------------------------------------------------------

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        resp = requests.post(
            self._search_endpoint,
            headers=self._build_headers(),
            json=self._build_request_body(query_bundle.query_str),
        )
        resp.raise_for_status()
        data = resp.json()
        return self._parse_results(data.get("results", []))

    # -- Async retrieval ------------------------------------------------------

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        async with aiohttp.ClientSession() as http_session:
            async with http_session.post(
                self._search_endpoint,
                headers=self._build_headers(),
                json=self._build_request_body(query_bundle.query_str),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        return self._parse_results(data.get("results", []))
