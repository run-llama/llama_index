from typing import Optional
from llama_index.core.base.response.schema import Response
from llama_index.core.bridge.pydantic import BaseModel
from pydantic import BaseModel as V2BaseModel


class ContributorQueryRequest(V2BaseModel):
    query: str


class ContributorQueryResponse(BaseModel):
    response: Optional[str]
    score: Optional[float]

    def __str__(self) -> str:
        """Convert to string representation."""
        return self.response or "None"

    def to_response(self) -> Response:
        """Convert to Response type."""
        return Response(response=self.response, metadata={"score": self.score})
