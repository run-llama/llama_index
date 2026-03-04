"""PlaygroundsSubgraphConnectorToolSpec."""

from typing import Optional, Union

import requests
from llama_index.tools.graphql.base import GraphQLToolSpec


class PlaygroundsSubgraphConnectorToolSpec(GraphQLToolSpec):
    """
    Connects to subgraphs on The Graph's decentralized network via the Playgrounds API.

    Attributes:
        spec_functions (list): List of functions that specify the tool's capabilities.
        url (str): The endpoint URL for the GraphQL requests.
        headers (dict): Headers used for the GraphQL requests.

    """

    spec_functions = ["graphql_request"]

    def __init__(self, identifier: str, api_key: str, use_deployment_id: bool = False):
        """
        Initialize the connector.

        Args:
            identifier (str): Subgraph identifier or Deployment ID.
            api_key (str): API key for the Playgrounds API.
            use_deployment_id (bool): Flag to indicate if the identifier is a deployment ID. Default is False.

        """
        endpoint = "deployments" if use_deployment_id else "subgraphs"
        self.url = (
            f"https://api.playgrounds.network/v1/proxy/{endpoint}/id/{identifier}"
        )
        self.headers = {
            "Content-Type": "application/json",
            "Playgrounds-Api-Key": api_key,
        }

    def graphql_request(
        self,
        query: str,
        variables: Optional[dict] = None,
        operation_name: Optional[str] = None,
    ) -> Union[dict, str]:
        """
        Make a GraphQL query.

        Args:
            query (str): The GraphQL query string to execute.
            variables (dict, optional): Variables for the GraphQL query. Default is None.
            operation_name (str, optional): Name of the operation, if multiple operations are present in the query. Default is None.

        Returns:
            dict: The response from the GraphQL server if successful.
            str: Error message if the request fails.

        """
        payload = {"query": query.strip()}

        if variables:
            payload["variables"] = variables

        if operation_name:
            payload["operationName"] = operation_name

        try:
            response = requests.post(self.url, headers=self.headers, json=payload)

            # Check if the request was successful
            response.raise_for_status()

            # Return the JSON response
            return response.json()

        except requests.RequestException as e:
            # Handle request errors
            return str(e)
        except ValueError as e:
            # Handle JSON decoding errors
            return f"Error decoding JSON: {e}"
