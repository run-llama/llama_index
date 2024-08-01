# Required Environment Variables: OPENAI_API_KEY

from llama_index.core.llama_pack import download_llama_pack
from llama_index.core import SimpleDirectoryReader

# download and install dependencies
HybridFusionRetrieverPack = download_llama_pack(
    "HybridFusionRetrieverPack", "./hybrid_fusion_pack"
)

# load documents
documents = SimpleDirectoryReader("./data").load_data()

# create the pack
hybrid_fusion_pack = HybridFusionRetrieverPack(
    documents, chunk_size=256, vector_similarity_top_k=2, bm25_similarity_top_k=2
)

# run the pack
response = hybrid_fusion_pack.run("Physical Standards for Letters?")
