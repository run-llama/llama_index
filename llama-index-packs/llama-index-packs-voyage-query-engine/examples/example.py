# Required Environment Variables: OPENAI_API_KEY, VOYAGE_API_KEY

from pathlib import Path
from llama_index.core.readers import download_loader
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
VoyageQueryEnginePack = download_llama_pack("VoyageQueryEnginePack", "./voyage_pack")
PDFReader = download_loader("PDFReader")

# load documents
loader = PDFReader()
document_path = Path("./data/101.pdf")  # replace with your own document
documents = loader.load_data(file=document_path)

# create the pack
voyage_pack = VoyageQueryEnginePack(documents)

# run the pack
response = voyage_pack.run("How to rewrite history?", similarity_top_k=2)
print(response)
