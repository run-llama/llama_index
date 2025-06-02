from typing import Any, Optional
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.networks.schema.contributor import (
    ContributorRetrieverRequest,
)
from llama_index.core.bridge.pydantic import Field, BaseModel, PrivateAttr
from llama_index.core.bridge.pydantic_settings import BaseSettings, SettingsConfigDict
from fastapi import FastAPI


class ContributorRetrieverServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=[".env", ".env.contributor.service"])
    api_version: str = Field(default="v1", description="API version.")
    secret: Optional[str] = Field(
        default=None, description="JWT secret."
    )  # left for future consideration.
    # or if user wants to implement their own


class ContributorRetrieverService(BaseModel):
    retriever: Optional[BaseRetriever]
    config: ContributorRetrieverServiceSettings
    _fastapi: FastAPI = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, retriever, config) -> None:
        super().__init__(retriever=retriever, config=config)
        self._fastapi = FastAPI(
            version=config.api_version,
        )

        # routes
        self._fastapi.add_api_route(path="/api", endpoint=self.index, methods=["GET"])
        self._fastapi.add_api_route(
            path="/api/retrieve",
            endpoint=self.retrieve,
            methods=["POST"],
        )

    async def index(self):
        """Index endpoint logic."""
        return {"message": "Hello World!"}

    async def retrieve(self, request: ContributorRetrieverRequest):
        """Retriever endpoint logic."""
        result = await self.retriever.aretrieve(request.query)
        return {
            "nodes_dict": result,
        }

    @classmethod
    def from_config_file(
        cls, env_file: str, retriever: BaseRetriever
    ) -> "ContributorRetrieverService":
        config = ContributorRetrieverServiceSettings(_env_file=env_file)
        return cls(retriever=retriever, config=config)

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
