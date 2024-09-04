from typing import Any, Optional
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.networks.schema.contributor import (
    ContributorQueryRequest,
)
from llama_index.core.bridge.pydantic import Field, BaseModel, PrivateAttr
from llama_index.core.bridge.pydantic_settings import BaseSettings, SettingsConfigDict
from fastapi import FastAPI


class ContributorQueryEngineServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=[".env", ".env.contributor.service"])
    api_version: str = Field(default="v1", description="API version.")
    secret: Optional[str] = Field(
        default=None, description="JWT secret."
    )  # left for future consideration.
    # or if user wants to implement their own


class ContributorQueryEngineService(BaseModel):
    query_engine: Optional[BaseQueryEngine]
    config: ContributorQueryEngineServiceSettings
    _fastapi: FastAPI = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, query_engine, config) -> None:
        super().__init__(query_engine=query_engine, config=config)
        self._fastapi = FastAPI(
            version=config.api_version,
        )

        # routes
        self._fastapi.add_api_route(path="/api", endpoint=self.index, methods=["GET"])
        self._fastapi.add_api_route(
            path="/api/query",
            endpoint=self.query,
            methods=["POST"],
        )

    async def index(self):
        """Index endpoint logic."""
        return {"message": "Hello World!"}

    async def query(self, request: ContributorQueryRequest):
        """Query endpoint logic."""
        result = await self.query_engine.aquery(request.query)
        return {
            "response": result.response,
            "source_nodes": result.source_nodes,
            "metadata": result.metadata,
        }

    @classmethod
    def from_config_file(
        cls, env_file: str, query_engine: BaseQueryEngine
    ) -> "ContributorQueryEngineService":
        config = ContributorQueryEngineServiceSettings(_env_file=env_file)
        return cls(query_engine=query_engine, config=config)

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__private_attributes__ or attr in self.model_fields:
            return super().__getattr__(attr)
        else:
            try:
                return getattr(self._fastapi, attr)
            except KeyError:
                raise AttributeError(
                    f"'{self.__class__.__name__}' fastapi app has no attribute '{attr}'"
                )

    @property
    def app(self):
        return self._fastapi


# keep for backwards compatibility
ContributorService = ContributorQueryEngineService
ContributorServiceSettings = ContributorQueryEngineServiceSettings
