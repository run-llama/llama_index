"""Amazon Neptune Database graph store index."""

import logging
from typing import Any, Dict, Optional

from .base import NeptuneBaseGraphStore, NeptuneQueryException
import boto3
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
        try:
            if credentials_profile_name is not None:
                session = boto3.Session(profile_name=credentials_profile_name)
            else:
                # use default credentials
                session = boto3.Session()

            client_params = {}
            if region_name:
                client_params["region_name"] = region_name

            protocol = "https" if use_https else "http"

            client_params["endpoint_url"] = f"{protocol}://{host}:{port}"

            if sign:
                self._client = session.client("neptunedata", **client_params)
            else:
                from botocore import UNSIGNED
                from botocore.config import Config

                self._client = session.client(
                    "neptunedata",
                    **client_params,
                    config=Config(signature_version=UNSIGNED),
                )

        except ImportError:
            raise ModuleNotFoundError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        except Exception as e:
            if type(e).__name__ == "UnknownServiceError":
                raise ModuleNotFoundError(
                    "Neptune Database requires a boto3 version 1.34.40 or greater."
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
