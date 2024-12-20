# Required Environment Variables: OPENAI_API_KEY

from llama_index.core.llama_pack import download_llama_pack
from llama_index.core import SimpleDirectoryReader

# download and install dependencies
QueryRewritingRetrieverPack = download_llama_pack(
    "QueryRewritingRetrieverPack", "./query_rewriting_pack"
)

# load documents
documents = SimpleDirectoryReader("./data").load_data()

# create the pack
query_rewriting_pack = QueryRewritingRetrieverPack(
    documents,
    chunk_size=256,
    vector_similarity_top_k=2,
)

# run the pack
response = query_rewriting_pack.run("Physical Standards for Letters?")
print(response)
