import os
from dotenv import load_dotenv
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore


load_dotenv()

index = PineconeVectorStore.from_params(
    api_key=os.getenv("PINECONE_API_KEY"),
    index_name="moonlit-ai-services-prod",
    namespace="document_details",
)

doc_id_filter = MetadataFilters(
    filters=[
        MetadataFilter(
            key="doc_id",
            value="02023R1803-20250101",
            operator=FilterOperator.EQ,
        )
    ],
)

nodes = index.get_nodes(filters=doc_id_filter, include_values=True)[:5]
for node in nodes:
    print(
        f"nodeid={node.node_id}, ref_doc_id={node.ref_doc_id}, embeddings={node.embedding}"
    )
