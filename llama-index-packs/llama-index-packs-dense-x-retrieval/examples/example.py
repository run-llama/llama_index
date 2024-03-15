# Required Environment Variables: OPENAI_API_KEY

from llama_index.core import SimpleDirectoryReader
from llama_index.packs.dense_x_retrieval import DenseXRetrievalPack

# load documents
documents = SimpleDirectoryReader("./data").load_data()

# uses the LLM to extract propositions from every document/node!
dense_pack = DenseXRetrievalPack(documents)

# run the pack
response = dense_pack.run("Physical Standards for Letters?")
print(response)
