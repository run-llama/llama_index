from typing import Any, Dict

import requests
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import QueryBundle


class CogniswitchQueryEngine(BaseQueryEngine):
    def __init__(self, cs_token: str, OAI_token: str, apiKey: str) -> None:
        """The required fields.

        Args:
            cs_token (str): Cogniswitch token.
            OAI_token (str): OpenAI token.
            apiKey (str): Oauth token.
        """
        self.cs_token = cs_token
        self.OAI_token = OAI_token
        self.apiKey = apiKey
        self.knowledge_request_endpoint = (
            "https://api.cogniswitch.ai:8243/cs-api/0.0.1/cs/knowledgeRequest"
        )
        self.headers = {
            "apiKey": self.apiKey,
            "platformToken": self.cs_token,
            "openAIToken": self.OAI_token,
        }

    def query_knowledge(self, query: str) -> Response:
        """
        Send a query to the Cogniswitch service and retrieve the response.

        Args:
            query (str): Query to be answered.

        Returns:
            dict: Response JSON from the Cogniswitch service.
        """
        data = {"query": query}
        response = requests.post(
            self.knowledge_request_endpoint,
            headers=self.headers,
            data=data,
        )
        if response.status_code == 200:
            resp = response.json()
            answer = resp["data"]["answer"]

            return Response(response=answer)
        else:
            error_message = response.json()["message"]
            return Response(response=error_message)

    def _query(self, query_bundle: QueryBundle) -> Response:
        return self.query_knowledge(query_bundle.query_str)

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        return self.query_knowledge(query_bundle.query_str)

    def _get_prompt_modules(self) -> Dict[str, Any]:
        """Get prompts."""
        return {}
