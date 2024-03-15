from pathlib import Path
from llama_index.readers.file.docs import PDFReader
from llama_index.packs.auto_merging_retriever import AutoMergingRetrieverPack

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
