import json
import os

from llama_index.node_parser.relational.dashscope import DashScopeJsonNodeParser
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import Document

os.environ['DASHSCOPE_API_KEY'] = 'sk-75878ade82164673a0962a825471e825'

doc_json = json.load(open('tests/documents.json'))
documents = []
for doc in doc_json:
    documents.append(Document.from_dict(doc))

node_parser = DashScopeJsonNodeParser(chunk_size=100, overlap_size=0, separator=' |,|，|。|？|！|\n|\?|\!')

pipeline = IngestionPipeline(
    transformations=[
        node_parser,
    ]
)

nodes = pipeline.run(documents=documents, show_progress=True)

for node in nodes:
    print(node)