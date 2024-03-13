from typing import Any, Optional
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.bridge.pydantic import Field, BaseModel
from llama_index.networks.schema.contributor import (
    ContributorRetrieverRequest,
)
from pydantic.v1 import BaseSettings, PrivateAttr
from fastapi import FastAPI


class ContributorRetrieverServiceSettings(BaseSettings):
    api_version: str = Field(default="v1", description="API version.")
    secret: Optional[str] = Field(
        default=None, description="JWT secret."
    )  # left for future consideration.
    # or if user wants to implement their own

    class Config:
        env_file = ".env", ".env.contributor.service"


class ContributorRetrieverService(BaseModel):
    retriever: Optional[BaseRetriever]
    config: ContributorRetrieverServiceSettings
    _fastapi: FastAPI = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, retriever, config) -> None:
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

        super().__init__(retriever=retriever, config=config)

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

    def __getattr__(self, attr) -> Any:
        if hasattr(self._fastapi, attr):
            return getattr(self._fastapi, attr)
        else:
            raise AttributeError(f"{attr} not exist")

    @property
    def app(self):
        return self._fastapi
