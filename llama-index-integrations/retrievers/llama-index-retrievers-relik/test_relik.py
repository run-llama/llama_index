from llama_index.core import Document
from llama_index.retrievers.relik import RelikPathExtractor

extractor = RelikPathExtractor(
    model="relik-ie/relik-relation-extraction-small-wikipedia",
    relationship_confidence_threshold=0.5,
    skip_errors=True,
)

nodes = extractor([Document.example()], show_progress=True)
print("AT THE END", nodes[0].metadata)
