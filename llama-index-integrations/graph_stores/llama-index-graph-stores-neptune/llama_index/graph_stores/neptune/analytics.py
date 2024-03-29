"""Amazon Neptune Analytics graph store index."""

import json
import logging
from typing import Any, Dict, Optional
from .base import NeptuneBaseGraphStore, NeptuneQueryException

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
        try:
            if client is not None:
                self._client = client
            else:
                import boto3

                if credentials_profile_name is not None:
                    session = boto3.Session(profile_name=credentials_profile_name)
                else:
                    # use default credentials
                    session = boto3.Session()

                self.graph_identifier = graph_identifier

                if region_name:
                    self._client = session.client(
                        "neptune-graph", region_name=region_name
                    )
                else:
                    self._client = session.client("neptune-graph")

        except ImportError:
            raise ModuleNotFoundError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        except Exception as e:
            if type(e).__name__ == "UnknownServiceError":
                raise ModuleNotFoundError(
                    "NeptuneGraph requires a boto3 version 1.34.40 or greater."
                    "Please install it with `pip install -U boto3`."
                ) from e
            else:
                raise ValueError(
                    "Could not load credentials to authenticate with AWS client. "
                    "Please check that credentials in the specified "
                    "profile name are valid."
                ) from e

        try:
            self._refresh_schema()
        except Exception as e:
            logger.error(
                f"Could not retrieve schema for Neptune due to the following error: {e}"
            )
            self.schema = None

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
