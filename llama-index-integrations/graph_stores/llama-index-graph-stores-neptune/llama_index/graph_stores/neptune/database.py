"""Amazon Neptune Database graph store index."""

import logging
from typing import Any, Dict, Optional
from .neptune import create_neptune_database_client
from .base import NeptuneBaseGraphStore
from .neptune import NeptuneQueryException
import json

logger = logging.getLogger(__name__)


class NeptuneDatabaseGraphStore(NeptuneBaseGraphStore):
    def __init__(
        self,
        host: str,
        port: int = 8182,
        use_https: bool = True,
        client: Any = None,
        credentials_profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        sign: bool = True,
        node_label: str = "Entity",
        **kwargs: Any,
    ) -> None:
        """Create a new Neptune Database graph wrapper instance."""
        self.node_label = node_label
        self._client = create_neptune_database_client(
            host, port, client, credentials_profile_name, region_name, sign, use_https
        )

    def query(self, query: str, params: dict = {}) -> Dict[str, Any]:
        """Query Neptune database."""
        try:
            logger.debug(f"query() query: {query} parameters: {json.dumps(params)}")
            return self.client.execute_open_cypher_query(
                openCypherQuery=query, parameters=json.dumps(params)
            )["results"]
        except Exception as e:
            raise NeptuneQueryException(
                {
                    "message": "An error occurred while executing the query.",
                    "details": str(e),
                    "query": query,
                    "parameters": str(params),
                }
            )

    def _get_summary(self) -> Dict:
        try:
            response = self.client.get_propertygraph_summary()
        except Exception as e:
            raise NeptuneQueryException(
                {
                    "message": (
                        "Summary API is not available for this instance of Neptune,"
                        "ensure the engine version is >=1.2.1.0"
                    ),
                    "details": str(e),
                }
            )

        try:
            summary = response["payload"]["graphSummary"]
        except Exception:
            raise NeptuneQueryException(
                {
                    "message": "Summary API did not return a valid response.",
                    "details": response.content.decode(),
                }
            )
        else:
            return summary
