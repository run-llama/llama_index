# Required Environment Variables: OPENAI_API_KEY

from pathlib import Path
from llama_index.readers.file.docs import PDFReader
from llama_index.packs.rag_fusion_query_pipeline import RAGFusionPipelinePack
from llama_index.llms.openai import OpenAI

# load documents
loader = PDFReader()
document_path = Path("./data/101.pdf")  # replace with your own document
documents = loader.load_data(file=document_path)

# create the pack
pack = RAGFusionPipelinePack(documents, llm=OpenAI(model="gpt-3.5-turbo"))

# run the pack
response = pack.run(input="How to rewrite history?")
print(response)
