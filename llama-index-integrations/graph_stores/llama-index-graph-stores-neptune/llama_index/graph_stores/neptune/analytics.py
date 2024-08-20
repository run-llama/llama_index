"""Amazon Neptune Analytics graph store index."""

import json
import logging
from .neptune import create_neptune_analytics_client
from typing import Any, Dict, Optional
from .base import NeptuneBaseGraphStore
from .neptune import NeptuneQueryException

logger = logging.getLogger(__name__)


class NeptuneAnalyticsGraphStore(NeptuneBaseGraphStore):
    def __init__(
        self,
        graph_identifier: str,
        client: Any = None,
        credentials_profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        node_label: str = "Entity",
        **kwargs: Any,
    ) -> None:
        """Create a new Neptune Analytics graph wrapper instance."""
        self.node_label = node_label
        self._client = create_neptune_analytics_client(
            graph_identifier, client, credentials_profile_name, region_name
        )
        self.graph_identifier = graph_identifier

    def query(self, query: str, params: dict = {}) -> Dict[str, Any]:
        """Query Neptune Analytics graph."""
        try:
            logger.debug(f"query() query: {query} parameters: {json.dumps(params)}")
            resp = self.client.execute_query(
                graphIdentifier=self.graph_identifier,
                queryString=query,
                parameters=params,
                language="OPEN_CYPHER",
            )
            return json.loads(resp["payload"].read().decode("UTF-8"))["results"]
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
            response = self.client.get_graph_summary(
                graphIdentifier=self.graph_identifier, mode="detailed"
            )
        except Exception as e:
            raise NeptuneQueryException(
                {
                    "message": ("Summary API error occurred on Neptune Analytics"),
                    "details": str(e),
                }
            )

        try:
            summary = response["graphSummary"]
        except Exception:
            raise NeptuneQueryException(
                {
                    "message": "Summary API did not return a valid response.",
                    "details": response.content.decode(),
                }
            )
        else:
            return summary
