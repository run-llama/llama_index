# Required Environment Variables: OPENAI_API_KEY

from pathlib import Path
from llama_index import download_loader
from llama_index.llama_pack import download_llama_pack
from llama_index.llms.openai import OpenAI

# download and install dependencies
RAGFusionPipelinePack = download_llama_pack(
    "RAGFusionPipelinePack", "./rag_fusion_pipeline_pack"
)
PDFReader = download_loader("PDFReader")

# load documents
loader = PDFReader()
document_path = Path("./data/101.pdf")  # replace with your own document
documents = loader.load_data(file=document_path)

# create the pack
pack = RAGFusionPipelinePack(documents, llm=OpenAI(model="gpt-3.5-turbo"))

# run the pack
response = pack.run(input="How to rewrite history?")
print(response)
