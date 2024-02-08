import os
from typing import Any, Dict, List

from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.schema import Document
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.openai import OpenAI


class VoyageQueryEnginePack(BaseLlamaPack):
    def __init__(self, documents: List[Document]) -> None:
        llm = OpenAI(model="gpt-4")
        embed_model = VoyageEmbedding(
            model_name="voyage-01", voyage_api_key=os.environ["VOYAGE_API_KEY"]
        )
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
        self.llm = llm
        self.index = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {"llm": self.llm, "index": self.index}

    def run(self, query_str: str, **kwargs: Any) -> Any:
        """Run the pipeline."""
        query_engine = self.index.as_query_engine(**kwargs)
        return query_engine.query(query_str)
