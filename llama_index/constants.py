"""Set of constants."""

DEFAULT_CONTEXT_WINDOW = 3900  # tokens
DEFAULT_NUM_OUTPUTS = 256  # tokens

DEFAULT_CHUNK_SIZE = 1024  # tokens
DEFAULT_CHUNK_OVERLAP = 20  # tokens
DEFAULT_SIMILARITY_TOP_K = 2

# NOTE: for text-embedding-ada-002
DEFAULT_EMBEDDING_DIM = 1536

# context window size for llm predictor
COHERE_CONTEXT_WINDOW = 2048
AI21_J2_CONTEXT_WINDOW = 8192


TYPE_KEY = "__type__"
DATA_KEY = "__data__"
VECTOR_STORE_KEY = "vector_store"
GRAPH_STORE_KEY = "graph_store"
INDEX_STORE_KEY = "index_store"
DOC_STORE_KEY = "doc_store"
