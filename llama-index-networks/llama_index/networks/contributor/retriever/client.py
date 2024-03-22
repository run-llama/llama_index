from typing import Optional, Dict, List
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.networks.schema.contributor import ContributorRetrieverResponse
from pydantic.v1 import BaseSettings, Field
import requests
import aiohttp


class ContributorRetrieverClientSettings(BaseSettings):
    """Settings for contributor."""

    api_key: Optional[str] = Field(default=None, env="API_KEY")
    api_url: str = Field(..., env="API_URL")

    class Config:
        env_file = ".env", ".env.contributor.client"


class ContributorRetrieverClient(BaseRetriever):
    """A remote Retriever exposed through a REST API."""

    def __init__(
        self,
        callback_manager: Optional[CallbackManager],
        config: ContributorRetrieverClientSettings,
    ) -> None:
        self.config = config
        super().__init__(callback_manager)

    @classmethod
    def from_config_file(
        cls, env_file: str, callback_manager: Optional[CallbackManager] = None
    ) -> "ContributorRetrieverClient":
        """Convenience constructor from a custom env file."""
        config = ContributorRetrieverClientSettings(_env_file=env_file)
        return cls(callback_manager=callback_manager, config=config)

    def _retrieve(
        self,
        query_bundle: QueryBundle,
        additional_data: Dict[str, str] = {},
        headers: Dict[str, str] = {},
    ) -> List[NodeWithScore]:
        """Make a post request to submit a query to Retriever."""
        # headers = {"Authorization": f"Bearer {self.config.api_key}"}
        data = {"query": query_bundle.query_str, "api_key": self.config.api_key}
        data.update(additional_data)
        result = requests.post(
            self.config.api_url + "/api/retrieve", json=data, headers=headers
        )
        try:
            contributor_response = ContributorRetrieverResponse.parse_obj(result.json())
        except Exception as e:
            raise ValueError("Failed to parse response") from e
        return contributor_response.get_nodes()

    async def _aretrieve(
        self,
        query_bundle: QueryBundle,
        api_token: Optional[str] = None,
        additional_data: Dict[str, str] = {},
        headers: Dict[str, str] = {},
    ) -> List[NodeWithScore]:
        """Make a post request to submit a query to Retriever."""
        # headers = {"Authorization": f"Bearer {self.config.api_key}"}
        data = {"query": query_bundle.query_str, "api_token": api_token}
        data.update(additional_data)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.api_url + "/api/retrieve", json=data, headers=headers
            ) as resp:
                json_result = await resp.json()
            try:
                contributor_response = ContributorRetrieverResponse.parse_obj(
                    json_result
                )
            except Exception as e:
                raise ValueError("Failed to parse response") from e
        return contributor_response.get_nodes()

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {}
