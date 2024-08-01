"""GraphQL Tool."""

from typing import Optional

import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class GraphQLToolSpec(BaseToolSpec):
    """Requests Tool."""

    spec_functions = ["graphql_request"]

    def __init__(self, url: str, headers: Optional[dict] = {}):
        self.headers = headers
        self.url = url

    def graphql_request(self, query: str, variables: str, operation_name: str):
        r"""
        Use this tool to make a GraphQL query against the server.

        Args:
            query (str): The GraphQL query to execute
            variables (str): The variable values for the query
            operation_name (str): The name for the query

        example input:
            "query":"query Ships {\n  ships {\n    id\n    model\n    name\n    type\n    status\n  }\n}",
            "variables":{},
            "operation_name":"Ships"

        """
        res = requests.post(
            self.url,
            headers=self.headers,
            json={
                "query": query,
                "variables": variables,
                "operationName": operation_name,
            },
        )
        return res.text
