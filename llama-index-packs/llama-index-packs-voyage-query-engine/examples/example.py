# Required Environment Variables: OPENAI_API_KEY, VOYAGE_API_KEY

from pathlib import Path
from llama_index.readers.file.docs import PDFReader
from llama_index.packs.voyage_query_engine import VoyageQueryEnginePack

# load documents
loader = PDFReader()
document_path = Path("./data/101.pdf")  # replace with your own document
documents = loader.load_data(file=document_path)

# create the pack
voyage_pack = VoyageQueryEnginePack(documents)

# run the pack
response = voyage_pack.run("Physical Standards for Letters", similarity_top_k=2)
print(response)
