from typing import Optional, Dict
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import QueryBundle
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.networks.schema.contributor import ContributorQueryResponse
from llama_index.core.bridge.pydantic_settings import BaseSettings, SettingsConfigDict
from llama_index.core.bridge.pydantic import Field
import requests
import aiohttp


class ContributorQueryEngineClientSettings(BaseSettings):
    """Settings for contributor."""

    model_config = SettingsConfigDict(env_file=[".env", ".env.contributor.client"])
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    api_url: str = Field(..., env="API_URL")


class ContributorQueryEngineClient(BaseQueryEngine):
    """A remote QueryEngine exposed through a REST API."""

    def __init__(
        self,
        callback_manager: Optional[CallbackManager],
        config: ContributorQueryEngineClientSettings,
    ) -> None:
        self.config = config
        super().__init__(callback_manager)

    @classmethod
    def from_config_file(
        cls, env_file: str, callback_manager: Optional[CallbackManager] = None
    ) -> "ContributorQueryEngineClient":
        """Convenience constructor from a custom env file."""
        config = ContributorQueryEngineClientSettings(_env_file=env_file)
        return cls(callback_manager=callback_manager, config=config)

    def _query(
        self,
        query_bundle: QueryBundle,
        additional_data: Dict[str, str] = {},
        headers: Dict[str, str] = {},
    ) -> RESPONSE_TYPE:
        """Make a post request to submit a query to QueryEngine."""
        # headers = {"Authorization": f"Bearer {self.config.api_key}"}
        data = {"query": query_bundle.query_str, "api_key": self.config.api_key}
        data.update(additional_data)
        result = requests.post(
            self.config.api_url + "/api/query", json=data, headers=headers
        )
        try:
            contributor_response = ContributorQueryResponse.model_validate(
                result.json()
            )
        except Exception as e:
            raise ValueError("Failed to parse response") from e
        return contributor_response.to_response()

    async def _aquery(
        self,
        query_bundle: QueryBundle,
        api_token: Optional[str] = None,
        additional_data: Dict[str, str] = {},
        headers: Dict[str, str] = {},
    ) -> RESPONSE_TYPE:
        """Make a post request to submit a query to QueryEngine."""
        # headers = {"Authorization": f"Bearer {self.config.api_key}"}
        data = {"query": query_bundle.query_str, "api_token": api_token}
        data.update(additional_data)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.api_url + "/api/query", json=data, headers=headers
            ) as resp:
                json_result = await resp.json()
            try:
                contributor_response = ContributorQueryResponse.model_validate(
                    json_result
                )
            except Exception as e:
                raise ValueError("Failed to parse response") from e
        return contributor_response.to_response()

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {}


# keep for backwards compatibility
ContributorClient = ContributorQueryEngineClient
ContributorClientSettings = ContributorQueryEngineClientSettings
