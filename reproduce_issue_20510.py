
import sys
from unittest.mock import MagicMock

from llama_index.core import Document, DocumentSummaryIndex
from llama_index.core.llms import MockLLM
from llama_index.core.embeddings import MockEmbedding

def reproduce():
    print("Starting reproduction...")
    
    # Mock dependencies
    llm = MockLLM()
    embed_model = MockEmbedding(embed_dim=1024)
    
    docs = [Document(text="Hello world")]
    
    print("Initializing DocumentSummaryIndex with show_progress=True...")
    try:
        index = DocumentSummaryIndex.from_documents(
            docs,
            llm=llm,
            embed_model=embed_model,
            show_progress=True
        )
        print("Success! Index created.")
    except TypeError as e:
        print(f"Caught expected TypeError: {e}")
        # Check if it matches the issue description
        if "unexpected keyword argument 'show_progress'" in str(e):
            print("Issue REPRODUCED.")
        else:
            print("Caught TypeError but message differs.")
    except Exception as e:
        print(f"Caught unexpected exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    reproduce()
