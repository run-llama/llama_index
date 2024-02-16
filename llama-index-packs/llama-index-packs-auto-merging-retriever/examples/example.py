from pathlib import Path
from llama_index.core.readers import download_loader
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
AutoMergingRetrieverPack = download_llama_pack(
    "AutoMergingRetrieverPack", "./auto_merging_retriever_pack"
)
PDFReader = download_loader("PDFReader")

# get documents from any data loader
loader = PDFReader()
document_path = Path("./data/101.pdf")  # replace with your own document
documents = loader.load_data(file=document_path)

# create the pack
auto_merging_retriever_pack = AutoMergingRetrieverPack(
    documents,
)

# run the pack
response = auto_merging_retriever_pack.run("Physical Standards for Letters")
print(response)
