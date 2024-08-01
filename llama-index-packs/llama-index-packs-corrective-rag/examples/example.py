# Required Environment Variables: OPENAI_API_KEY
# Required TavilyAI API KEY for web searches - https://tavily.com/
from llama_index.core import SimpleDirectoryReader
from llama_index.packs.corrective_rag import CorrectiveRAGPack

# load documents
documents = SimpleDirectoryReader("./data").load_data()

# uses the LLM to extract propositions from every document/node!
corrective_rag = CorrectiveRAGPack(documents, tavily_ai_apikey="<tavily_ai_apikey>")

# run the pack
response = corrective_rag.run("<Query>")
print(response)
